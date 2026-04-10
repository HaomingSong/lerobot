#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class EO1VisionFlowMatchingOutputWithPast(ModelOutput):
    fm_loss: torch.FloatTensor | None = None
    # logits: torch.FloatTensor | None = None
    # past_key_values: list[torch.FloatTensor] | None = None
    # hidden_states: torch.FloatTensor | None = None
    # attentions: torch.FloatTensor | None = None
    # rope_deltas: torch.LongTensor | None = None


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


class EO1Policy(PreTrainedPolicy):
    """EO1 policy wrapper for LeRobot robot-only training/evaluation."""

    config_class = EO1Config
    name = "eo1"

    def __init__(self, config: EO1Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        resolved_attn_implementation = config.resolved_attn_implementation

        if resolved_attn_implementation != config.attn_implementation:
            logger.warning(
                "EO1 torch.compile is not compatible with flash_attention_2; falling back to %s.",
                resolved_attn_implementation,
            )

        if config.pretrained_path is None:
            # Initialize from pretrained VLM
            vlm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.vlm_base,
                dtype=config.dtype,
                attn_implementation=resolved_attn_implementation,
            )
        else:
            vlm_backbone = Qwen2_5_VLForConditionalGeneration(config.vlm_backbone_config)

        self.model = EO1VisionFlowMatchingModel(config, vlm_backbone)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        state = self.prepare_state(batch.get(OBS_STATE))
        actions = self.prepare_action(batch.get(ACTION))
        model_inputs = {k: v for k, v in batch.items() if k not in {OBS_STATE, ACTION}}
        outputs = self.model(states=state, action=actions, **model_inputs)

        loss = outputs.fm_loss

        loss_dict = {"loss": loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        # Prepare inputs
        states = self.prepare_state(batch.get(OBS_STATE))
        model_inputs = {k: v for k, v in batch.items() if k != OBS_STATE}
        actions = self.model.sample_actions(states=states, **model_inputs).to(torch.float32)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def prepare_state(self, state):
        """Pad state"""
        if state is None:
            return None
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, action):
        """Pad action"""
        if action is None:
            return None
        actions = pad_vector(action, self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def get_optim_params(self) -> dict:
        return self.parameters()


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


class EO1VisionActionProjector(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_layer: str = "linear",
        bias: bool = True,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ):
        layers = []
        in_dim = in_channels
        hidden_channels = [in_dim] * (num_layers - 1) + [out_channels]
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device))
            layers.append(ACT2FN[activation_layer])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=dtype, device=device))
        super().__init__(*layers)

    @property
    def dtype(self):
        return self[0].weight.dtype


class EO1VisionFlowMatchingModel(nn.Module):
    def __init__(
        self,
        config: EO1Config,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration | None = None,
    ):
        super().__init__()

        self.config = config
        self.vlm_backbone = vlm_backbone.to(dtype=getattr(torch, config.dtype))
        self.hidden_size = self.vlm_backbone.config.text_config.hidden_size
        max_state_dim = config.max_state_dim
        max_action_dim = config.max_action_dim
        self.state_proj = nn.Linear(max_state_dim, self.hidden_size, dtype=torch.float32)
        self.action_in_proj = nn.Linear(max_action_dim, self.hidden_size, dtype=torch.float32)
        self.action_out_proj = EO1VisionActionProjector(
            self.hidden_size,
            max_action_dim,
            config.num_action_layers,
            config.action_act,
            dtype=torch.float32,
        )
        self.action_time_mlp_in = nn.Linear(self.hidden_size * 2, self.hidden_size, dtype=torch.float32)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float32)
        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.suppress_errors = True
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def get_input_embeddings(self):
        return self.vlm_backbone.get_input_embeddings()

    def flow_head_autocast_context(self):
        return torch.autocast(device_type=self.state_proj.weight.device.type, enabled=False)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the Qwen2.5-VL backbone."""
        self.gradient_checkpointing_enabled = True
        self.vlm_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Enabled gradient checkpointing for EO1VisionFlowMatchingModel")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the Qwen2.5-VL backbone."""
        self.gradient_checkpointing_enabled = False
        self.vlm_backbone.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for EO1VisionFlowMatchingModel")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Apply manual gradient checkpointing to EO1 flow-head computations when training."""
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def build_flow_matching_targets(
        self,
        action: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample diffusion targets while keeping padded actions deterministic.

        Padded action slots are excluded from the denoising loss, so they should not
        receive random noise injections that can still perturb the transformer context.
        """
        time = self.sample_time(action.shape[0], device)  # (b,)
        time_expanded = time[:, None, None].expand(-1, action.shape[1], 1).clone()  # (b, h, 1)

        pad_mask = None
        if action_is_pad is not None:
            pad_mask = action_is_pad.to(device=device, dtype=torch.bool).unsqueeze(-1)
            time_expanded = time_expanded.masked_fill(pad_mask, 0.0)

        noise = self.sample_noise(action.shape, device)
        x_t = time_expanded * noise + (1 - time_expanded) * action
        u_t = noise - action

        if pad_mask is not None:
            u_t = u_t.masked_fill(pad_mask, 0.0)

        return time, x_t, u_t

    @staticmethod
    def reduce_flow_matching_loss(
        losses: torch.Tensor,
        action_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_is_pad is None:
            return losses.mean()

        valid_mask = (~action_is_pad).reshape(-1, 1).to(device=losses.device, dtype=losses.dtype)
        valid_losses = losses * valid_mask
        valid_elements = valid_mask.sum() * losses.shape[-1]
        return valid_losses.sum() / valid_elements.clamp_min(1)

    def replace_special_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        special_features: torch.FloatTensor = None,
        special_token_ids: torch.LongTensor = None,
    ) -> torch.LongTensor:
        """Replace the special embeddings with the special features."""
        if special_features is not None and special_token_ids is not None:
            n_special_tokens = (input_ids == special_token_ids).sum().item()
            n_special_features = special_features.shape[0]
            assert n_special_tokens == n_special_features, (
                f"Special features and special tokens {special_token_ids} do not match: \
                tokens: {n_special_tokens}, features {n_special_features}"
            )
            mask = input_ids == special_token_ids
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            special_mask = mask_expanded.to(inputs_embeds.device)
            special_features = special_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_mask, special_features)
        return inputs_embeds, None

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        state_token_id: int | None = None,
    ) -> torch.FloatTensor:
        """Embed the suffix"""
        if inputs_embeds is None:

            def input_embed_func(input_ids: torch.LongTensor) -> torch.FloatTensor:
                return self.get_input_embeddings()(input_ids)

            inputs_embeds = self._apply_checkpoint(input_embed_func, input_ids)

        if pixel_values is not None:

            def image_embed_func(
                pixel_values: torch.Tensor,
                image_grid_thw: torch.LongTensor | None,
            ) -> torch.FloatTensor:
                image_embeds = self.vlm_backbone.get_image_features(pixel_values, image_grid_thw)
                return torch.cat(image_embeds, dim=0)

            image_embeds = self._apply_checkpoint(image_embed_func, pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.vlm_backbone.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if states is not None:

            def state_proj_func(states: torch.Tensor) -> torch.FloatTensor:
                with self.flow_head_autocast_context():
                    states = states.to(dtype=torch.float32)
                    return self.state_proj(states)

            state_embs = self._apply_checkpoint(state_proj_func, states)
            inputs_embeds, _ = self.replace_special_embeddings(
                input_ids, inputs_embeds, state_embs, state_token_id
            )
        return inputs_embeds

    def embed_suffix(
        self,
        timestep: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.FloatTensor:
        """Embed the suffix"""

        def action_proj_func(noisy_actions: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
                return self.action_in_proj(noisy_actions)

        action_embs = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_embs = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            device=action_embs.device,
        )
        time_embs = time_embs.to(dtype=action_embs.dtype)
        time_embs = time_embs[:, None, :].expand_as(action_embs)
        action_time_embs = torch.cat([action_embs, time_embs], dim=2)

        def mlp_func(action_time_embs: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                action_time_embs = self.action_time_mlp_in(action_time_embs)
                action_time_embs = F.silu(action_time_embs)
                return self.action_time_mlp_out(action_time_embs)

        action_time_embs = self._apply_checkpoint(mlp_func, action_time_embs)
        return action_time_embs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        state_token_id: int = 151669,
        action_token_id: int = 151666,
        **kwargs,
    ) -> EO1VisionFlowMatchingOutputWithPast:
        """multi-modal forward pass, including image, video, state, action, and language."""
        inputs_embeds = self.embed_prefix(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            states=states,
            state_token_id=state_token_id,
        )

        if action is not None:
            noise_mask = input_ids == action_token_id
            mask = noise_mask

            time, x_t, u_t = self.build_flow_matching_targets(action, action_is_pad, inputs_embeds.device)

            action_time_embs = self.embed_suffix(time, x_t)
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
            action_mask = mask_expanded.to(inputs_embeds.device)

            action_time_embs = action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_time_embs)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # Training only needs the final hidden states for the action tokens.
        # Avoid the CausalLM wrapper here so we do not materialize lm_head logits,
        # full-layer hidden states, or KV cache during every train step.
        position_ids, _ = self.vlm_backbone.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        def vlm_forward_func(
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            inputs_embeds: torch.FloatTensor,
        ) -> torch.FloatTensor:
            outputs = self.vlm_backbone.model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

        hidden_states = self._apply_checkpoint(vlm_forward_func, position_ids, attention_mask, inputs_embeds)

        fm_loss = None
        v_t = None
        if action is not None:
            action_hidden_states = hidden_states[action_mask[..., 0]]

            def action_out_proj_func(action_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
                with self.flow_head_autocast_context():
                    action_hidden_states = action_hidden_states.to(dtype=self.action_out_proj.dtype)
                    return self.action_out_proj(action_hidden_states)

            v_t = self._apply_checkpoint(action_out_proj_func, action_hidden_states)
            u_t = u_t.reshape(v_t.shape)
            v_t = v_t.to(dtype=u_t.dtype)

            losses = F.mse_loss(u_t, v_t, reduction="none")
            fm_loss = self.reduce_flow_matching_loss(losses, action_is_pad)

        return EO1VisionFlowMatchingOutputWithPast(
            fm_loss=fm_loss,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        state_token_id: int = 151669,
        action_token_id: int = 151666,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""
        chunk_size = self.config.chunk_size

        action_mask = input_ids == action_token_id
        if action_mask.ne(action_mask[:1]).any():
            raise ValueError(
                "Batch inference expects all samples to share the same action token mask after left padding."
            )

        act_start = int(action_mask[0].to(torch.int64).argmax().item())
        act_end = act_start + chunk_size
        act_slice = slice(act_start, act_end)

        position_ids, _ = self.vlm_backbone.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.embed_prefix(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            states=states,
            state_token_id=state_token_id,
        ).clone()

        batch_size = input_ids.shape[0]
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        outputs = self.vlm_backbone.model(
            position_ids=position_ids[..., :act_start],
            attention_mask=attention_mask[:, :act_start],
            inputs_embeds=inputs_embeds[:, :act_start],
            use_cache=True,
        )

        x_t = self.sample_noise(
            (batch_size, chunk_size, self.config.max_action_dim),
            device,
        ).to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.config.num_denoise_steps
        past_key_values = outputs.past_key_values

        for step in range(self.config.num_denoise_steps):
            time = torch.full(
                (batch_size,),
                1.0 + step * dt,
                device=device,
                dtype=torch.float32,
            )
            action_time_embs = self.embed_suffix(time, x_t)
            inputs_embeds[:, act_slice] = action_time_embs.to(inputs_embeds.dtype)

            outputs = self.vlm_backbone.model(
                position_ids=position_ids[..., act_slice],
                attention_mask=attention_mask[:, :act_end],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, act_slice],
                use_cache=True,
            )
            with self.flow_head_autocast_context():
                hidden_states = outputs.last_hidden_state[:, :chunk_size].to(dtype=self.action_out_proj.dtype)
                v_t = self.action_out_proj(hidden_states)

            x_t += dt * v_t.reshape(x_t.shape)

        return x_t
