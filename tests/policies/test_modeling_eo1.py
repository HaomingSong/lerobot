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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.eo1.modeling_eo1 import EO1Policy, EO1VisionFlowMatchingModel

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QWEN_ROOT = WORKSPACE_ROOT / "eo1_artifacts" / "Qwen2.5-VL-3B-Instruct"


class DummyVLMBackbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(512, hidden_size)
        self.rope_deltas = None
        self.forward_calls: list[dict[str, object]] = []
        self.model_calls: list[dict[str, object]] = []

    def get_input_embeddings(self):
        return self.embedding

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).view(1, 1, -1).expand(3, batch_size, -1)
        return position_ids, None

    def model(
        self,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        self.model_calls.append(
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "use_cache": use_cache,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=SimpleNamespace(crop=lambda prefix_len: None),
        )

    def forward(
        self,
        *,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
        **kwargs,
    ):
        self.forward_calls.append(
            {
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "image_grid_thw": image_grid_thw,
                "logits_to_keep": logits_to_keep,
            }
        )
        return SimpleNamespace(
            hidden_states=inputs_embeds,
            logits=None,
            past_key_values=None,
            attentions=None,
        )


def make_test_config(**overrides) -> EO1Config:
    qwen_root = Path(DEFAULT_QWEN_ROOT)
    if not qwen_root.exists():
        pytest.skip(f"EO1 Qwen checkpoint not found: {qwen_root}")
    return EO1Config(vlm_base=str(qwen_root), **overrides)


def test_eo1_bfloat16_backbone_and_fp32_flow_head():
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    assert model.vlm_backbone.embedding.weight.dtype == torch.bfloat16
    assert model.state_proj.weight.dtype == torch.float32
    assert model.action_in_proj.weight.dtype == torch.float32
    assert model.action_time_mlp_in.weight.dtype == torch.float32
    assert model.action_time_mlp_out.weight.dtype == torch.float32
    assert model.action_out_proj[0].weight.dtype == torch.float32


def test_eo1_state_proj_uses_max_state_dim():
    config = make_test_config(dtype="bfloat16", max_state_dim=24, max_action_dim=32)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    assert model.state_proj.in_features == config.max_state_dim
    assert model.action_in_proj.in_features == config.max_action_dim


def test_eo1_prepare_helpers_pad_without_mutating_batch_contract():
    policy = SimpleNamespace(config=SimpleNamespace(max_state_dim=6, max_action_dim=8))
    state = torch.ones(2, 4, dtype=torch.float32)
    action = torch.ones(2, 3, 5, dtype=torch.float32)

    padded_state = EO1Policy.prepare_state(policy, state)
    padded_action = EO1Policy.prepare_action(policy, action)

    assert padded_state.shape == (2, 6)
    assert padded_action.shape == (2, 3, 8)
    assert EO1Policy.prepare_state(policy, None) is None
    assert EO1Policy.prepare_action(policy, None) is None


def test_eo1_embed_suffix_stays_fp32_inside_global_autocast():
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    timestep = torch.tensor([0.5], dtype=torch.float32)
    noisy_actions = torch.randn(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        action_time_embs = model.embed_suffix(timestep, noisy_actions)

    assert action_time_embs.dtype == torch.float32


def test_eo1_forward_loss_stays_fp32_inside_global_autocast():
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))
    action_token_id = 7

    input_ids = torch.full((1, config.chunk_size), action_token_id, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    action = torch.randn(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action=action,
            action_token_id=action_token_id,
        )

    assert outputs.fm_loss is not None
    assert outputs.fm_loss.dtype == torch.float32


def test_eo1_padded_actions_skip_diffusion_noise():
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    action = torch.zeros(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)
    action_is_pad = torch.zeros(1, config.chunk_size, dtype=torch.bool)
    action_is_pad[:, config.chunk_size // 2 :] = True

    model.sample_time = lambda bsize, device: torch.full((bsize,), 0.5, dtype=torch.float32, device=device)
    model.sample_noise = lambda shape, device: torch.full(shape, 4.0, dtype=torch.float32, device=device)

    captured: dict[str, torch.Tensor] = {}

    def fake_embed_suffix(timestep: torch.Tensor, noisy_actions: torch.Tensor) -> torch.Tensor:
        captured["timestep"] = timestep.detach().clone()
        captured["noisy_actions"] = noisy_actions.detach().clone()
        return torch.zeros(
            noisy_actions.shape[0],
            noisy_actions.shape[1],
            config.text_config.hidden_size,
            dtype=torch.float32,
            device=noisy_actions.device,
        )

    model.embed_suffix = fake_embed_suffix
    action_token_id = 7

    outputs = model(
        input_ids=torch.full((1, config.chunk_size), action_token_id, dtype=torch.long),
        attention_mask=torch.ones(1, config.chunk_size, dtype=torch.long),
        action=action,
        action_is_pad=action_is_pad,
        action_token_id=action_token_id,
    )

    assert outputs.fm_loss is not None
    assert torch.allclose(
        captured["noisy_actions"][0, : config.chunk_size // 2],
        torch.full((config.chunk_size // 2, config.max_action_dim), 2.0, dtype=torch.float32),
    )
    assert torch.count_nonzero(captured["noisy_actions"][0, config.chunk_size // 2 :]).item() == 0


def test_eo1_flow_loss_only_averages_valid_actions():
    losses = torch.tensor(
        [
            [1.0, 3.0],
            [10.0, 12.0],
        ],
        dtype=torch.float32,
    )
    action_is_pad = torch.tensor([[False, True]])

    reduced = EO1VisionFlowMatchingModel.reduce_flow_matching_loss(losses, action_is_pad)

    assert reduced.item() == pytest.approx(2.0)


def test_eo1_sample_actions_supports_batched_eval_denoising():
    config = make_test_config(dtype="bfloat16", num_denoise_steps=3)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    recorded_timesteps: list[torch.Tensor] = []
    original_embed_suffix = model.embed_suffix

    def capture_embed_suffix(timestep: torch.Tensor, noisy_actions: torch.Tensor) -> torch.Tensor:
        recorded_timesteps.append(timestep.detach().clone())
        return original_embed_suffix(timestep, noisy_actions)

    model.embed_suffix = capture_embed_suffix

    batch_size = 2
    action_token_id = 7
    state_token_id = 9
    input_ids = torch.cat(
        [
            torch.full((batch_size, 1), state_token_id, dtype=torch.long),
            torch.full((batch_size, config.chunk_size), action_token_id, dtype=torch.long),
            torch.zeros((batch_size, 1), dtype=torch.long),
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)
    states = torch.randn(batch_size, config.max_state_dim, dtype=torch.float32)

    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        states=states,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )

    assert actions.shape == (batch_size, config.chunk_size, config.max_action_dim)
    assert len(recorded_timesteps) == config.num_denoise_steps
    assert len(model.vlm_backbone.model_calls) == 1 + config.num_denoise_steps

    expected_times = [1.0, 2.0 / 3.0, 1.0 / 3.0]
    for timestep, expected in zip(recorded_timesteps, expected_times, strict=True):
        assert timestep.shape == (batch_size,)
        torch.testing.assert_close(
            timestep,
            torch.full((batch_size,), expected, dtype=torch.float32),
        )


def test_eo1_state_dict_keeps_flow_head_weights_in_fp32():
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    state_dict = model.state_dict()

    assert state_dict["vlm_backbone.embedding.weight"].dtype == torch.bfloat16
    for name, tensor in state_dict.items():
        if name.startswith(
            (
                "state_proj.",
                "action_in_proj.",
                "action_time_mlp_in.",
                "action_time_mlp_out.",
                "action_out_proj.",
            )
        ):
            assert tensor.dtype == torch.float32
