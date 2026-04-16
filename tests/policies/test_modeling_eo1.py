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

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.eo1.modeling_eo1 import EO1Policy, EO1VisionFlowMatchingModel
from lerobot.utils.constants import ACTION

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QWEN_ROOT = WORKSPACE_ROOT / "eo1_artifacts" / "Qwen2.5-VL-3B-Instruct"


class DummyVLMBackbone(nn.Module):
    def __init__(self, hidden_size: int, resolved_attn_implementation: str | None = None):
        super().__init__()
        self.embedding = nn.Embedding(512, hidden_size)
        self.config = SimpleNamespace(
            _attn_implementation=resolved_attn_implementation,
            text_config=SimpleNamespace(hidden_size=hidden_size),
        )
        self.rope_deltas = None
        self.forward_calls: list[dict[str, object]] = []
        self.model_calls: list[dict[str, object]] = []
        self.rope_index_calls: list[dict[str, object]] = []
        self.gradient_checkpointing = False
        self.gradient_checkpointing_enable_calls: list[dict[str, object] | None] = []
        self.gradient_checkpointing_disable_calls = 0
        self.image_feature_calls: list[dict[str, object]] = []
        self.video_feature_calls: list[dict[str, object]] = []

    @property
    def model(self):
        return self

    def get_input_embeddings(self):
        return self.embedding

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ):
        self.rope_index_calls.append(
            {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
                "second_per_grid_ts": second_per_grid_ts,
                "mm_token_type_ids": mm_token_type_ids,
            }
        )
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            text_positions = attention_mask.long().cumsum(-1) - 1
            text_positions = text_positions.masked_fill(attention_mask == 0, 0)
        else:
            text_positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        position_ids = text_positions.view(1, batch_size, seq_len).expand(3, batch_size, seq_len)
        rope_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=input_ids.device)
        return position_ids, rope_deltas

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object] | None = None):
        self.gradient_checkpointing = True
        self.gradient_checkpointing_enable_calls.append(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.gradient_checkpointing_disable_calls += 1

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        self.model_calls.append(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
                "use_cache": use_cache,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=SimpleNamespace(crop=lambda prefix_len: None),
        )


def make_test_config(**overrides) -> EO1Config:
    qwen_root = Path(DEFAULT_QWEN_ROOT)
    if not qwen_root.exists():
        pytest.skip(f"EO1 Qwen checkpoint not found: {qwen_root}")
    overrides.setdefault(
        "output_features",
        {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    )
    return EO1Config(vlm_base=str(qwen_root), **overrides)


def patch_checkpoint_to_record(monkeypatch):
    calls: list[str] = []

    def recording_checkpoint(func, *args, **kwargs):
        calls.append(func.__name__)
        kwargs.pop("use_reentrant", None)
        kwargs.pop("preserve_rng_state", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recording_checkpoint)
    return calls


def test_eo1_config_applies_requested_attn_implementation_without_mutating_vlm_config():
    config = make_test_config(attn_implementation="flash_attention_2")
    original_vlm_config = deepcopy(config.vlm_config)

    vlm_config = config.vlm_backbone_config

    assert isinstance(config.vlm_config, dict)
    assert vlm_config._attn_implementation == "flash_attention_2"
    assert vlm_config.text_config._attn_implementation == "flash_attention_2"
    assert vlm_config.vision_config._attn_implementation == "flash_attention_2"
    assert config.vlm_backbone_config._attn_implementation == "flash_attention_2"
    assert config.text_config._attn_implementation == "flash_attention_2"
    assert config.vision_config._attn_implementation == "flash_attention_2"
    assert config.vlm_config == original_vlm_config


def test_eo1_config_keeps_provided_vlm_config_without_reloading(monkeypatch):
    seed_config = make_test_config()
    custom_vlm_config = deepcopy(seed_config.vlm_config)
    custom_vlm_config["text_config"]["hidden_size"] = 1234

    def fail_from_pretrained(*args, **kwargs):
        raise AssertionError("from_pretrained should not be called when vlm_config is provided")

    monkeypatch.setattr(
        "lerobot.policies.eo1.configuration_eo1.Qwen2_5_VLConfig.from_pretrained",
        fail_from_pretrained,
    )

    config = EO1Config(vlm_base="unused", vlm_config=custom_vlm_config)

    assert config.vlm_config == custom_vlm_config
    assert config.vlm_backbone_config.text_config.hidden_size == 1234


def test_eo1_config_roundtrip_persists_requested_attn_implementation(tmp_path):
    config = make_test_config(attn_implementation="flash_attention_2")

    config._save_pretrained(tmp_path)
    reloaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(reloaded, EO1Config)
    assert reloaded.attn_implementation == "flash_attention_2"
    assert isinstance(reloaded.vlm_config, dict)
    assert reloaded.vlm_backbone_config._attn_implementation == "flash_attention_2"


def test_eo1_sample_time_uses_configured_distribution(monkeypatch):
    config = make_test_config(
        dtype="bfloat16",
        time_sampling_beta_alpha=2.0,
        time_sampling_beta_beta=3.0,
        time_sampling_scale=0.5,
        time_sampling_offset=0.25,
    )
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    captured: dict[str, object] = {}

    class FakeBeta:
        def __init__(self, concentration1, concentration0):
            captured["concentration1"] = concentration1
            captured["concentration0"] = concentration0

        def sample(self, shape):
            captured["shape"] = shape
            return torch.full(shape, 0.5, dtype=torch.float32)

    monkeypatch.setattr(torch.distributions, "Beta", FakeBeta)

    sampled_time = model.sample_time(3, torch.device("cpu"))

    assert captured == {
        "concentration1": 2.0,
        "concentration0": 3.0,
        "shape": (3,),
    }
    torch.testing.assert_close(sampled_time, torch.full((3,), 0.5, dtype=torch.float32))


def test_eo1_bfloat16_backbone_and_fp32_flow_head():
    config = make_test_config(dtype="bfloat16")
    backbone = DummyVLMBackbone(config.text_config.hidden_size).to(dtype=torch.bfloat16)
    model = EO1VisionFlowMatchingModel(config, backbone)

    assert model.hidden_size == config.text_config.hidden_size
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


def test_eo1_gradient_checkpointing_enable_propagates_to_backbone():
    config = make_test_config(dtype="bfloat16")
    backbone = DummyVLMBackbone(config.text_config.hidden_size)
    model = EO1VisionFlowMatchingModel(config, backbone)

    model.gradient_checkpointing_enable()

    assert model.gradient_checkpointing_enabled is True
    assert backbone.gradient_checkpointing is True
    assert backbone.gradient_checkpointing_enable_calls == [{"use_reentrant": False}]


def test_eo1_gradient_checkpointing_disable_propagates_to_backbone():
    config = make_test_config(dtype="bfloat16")
    backbone = DummyVLMBackbone(config.text_config.hidden_size)
    model = EO1VisionFlowMatchingModel(config, backbone)

    model.gradient_checkpointing_enable()
    model.gradient_checkpointing_disable()

    assert model.gradient_checkpointing_enabled is False
    assert backbone.gradient_checkpointing is False
    assert backbone.gradient_checkpointing_disable_calls == 1


def test_eo1_policy_enables_gradient_checkpointing_from_config(monkeypatch):
    config = make_test_config(dtype="bfloat16", gradient_checkpointing=True)
    config.input_features["observation.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))

    backbone = DummyVLMBackbone(config.text_config.hidden_size)

    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: backbone,
    )

    policy = EO1Policy(config)

    assert policy.model.gradient_checkpointing_enabled is True
    assert backbone.gradient_checkpointing is True
    assert backbone.gradient_checkpointing_enable_calls == [{"use_reentrant": False}]


def test_eo1_policy_forwards_attn_implementation_to_vlm_from_pretrained(monkeypatch):
    config = make_test_config(dtype="bfloat16", attn_implementation="flash_attention_2")
    config.input_features["observation.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))

    captured: dict[str, object] = {}

    def fake_from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return DummyVLMBackbone(
            config.text_config.hidden_size,
            resolved_attn_implementation=kwargs.get("attn_implementation"),
        )

    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        fake_from_pretrained,
    )

    EO1Policy(config)

    assert captured["args"] == (config.vlm_base,)
    assert captured["kwargs"] == {
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    }


def test_eo1_policy_applies_attn_implementation_to_constructed_vlm_config(monkeypatch):
    config = make_test_config(
        dtype="bfloat16",
        pretrained_path=Path("/tmp/fake-pretrained-policy"),
        attn_implementation="sdpa",
    )
    config.input_features["observation.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))

    captured: dict[str, object] = {}

    def fake_from_config(vlm_config, **kwargs):
        captured["vlm_config"] = vlm_config
        captured["kwargs"] = kwargs
        return DummyVLMBackbone(
            vlm_config.text_config.hidden_size,
            resolved_attn_implementation=vlm_config._attn_implementation,
        )

    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration._from_config",
        fake_from_config,
    )

    EO1Policy(config)

    vlm_config = captured["vlm_config"]
    assert captured["kwargs"] == {"dtype": "bfloat16"}
    assert vlm_config._attn_implementation == "sdpa"
    assert vlm_config.text_config._attn_implementation == "sdpa"
    assert vlm_config.vision_config._attn_implementation == "sdpa"


def test_eo1_policy_uses_backbone_default_dtype_for_auto_from_config(monkeypatch):
    config = make_test_config(
        dtype="auto",
        pretrained_path=Path("/tmp/fake-pretrained-policy"),
        attn_implementation="sdpa",
    )
    config.input_features["observation.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))

    captured: dict[str, object] = {}

    def fake_from_config(vlm_config, **kwargs):
        captured["vlm_config"] = vlm_config
        captured["kwargs"] = kwargs
        return DummyVLMBackbone(
            vlm_config.text_config.hidden_size,
            resolved_attn_implementation=vlm_config._attn_implementation,
        )

    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration._from_config",
        fake_from_config,
    )

    EO1Policy(config)

    assert captured["kwargs"] == {"dtype": config.vlm_backbone_config.dtype}


def test_eo1_apply_checkpoint_only_runs_when_training_and_grad_enabled(monkeypatch):
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))
    calls = patch_checkpoint_to_record(monkeypatch)
    model.gradient_checkpointing_enable()
    model.train()

    result = model._apply_checkpoint(lambda x: x + 1, torch.tensor(1.0))
    assert result.item() == pytest.approx(2.0)
    assert calls == ["<lambda>"]

    calls.clear()
    with torch.no_grad():
        result = model._apply_checkpoint(lambda x: x + 2, torch.tensor(1.0))

    assert result.item() == pytest.approx(3.0)
    assert calls == []


def test_eo1_prepare_helpers_pad_without_mutating_batch_contract():
    policy = SimpleNamespace(config=SimpleNamespace(max_state_dim=6, max_action_dim=8))
    state = torch.ones(2, 4, dtype=torch.float32)
    action = torch.ones(2, 3, 5, dtype=torch.float32)

    padded_state = EO1Policy.prepare_state(policy, state)
    padded_action = EO1Policy.prepare_action(policy, action)

    assert padded_state.shape == (2, 6)
    assert padded_action.shape == (2, 3, 8)


def test_eo1_embed_suffix_stays_fp32_inside_global_autocast():
    config = make_test_config(dtype="bfloat16", force_fp32_autocast=True)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    timestep = torch.tensor([0.5], dtype=torch.float32)
    noisy_actions = torch.randn(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        action_time_embs = model.embed_suffix(timestep, noisy_actions)

    assert action_time_embs.dtype == torch.float32


def test_eo1_embed_suffix_uses_manual_checkpointing(monkeypatch):
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))
    calls = patch_checkpoint_to_record(monkeypatch)
    model.gradient_checkpointing_enable()
    model.train()

    timestep = torch.tensor([0.5], dtype=torch.float32)
    noisy_actions = torch.randn(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)

    action_time_embs = model.embed_suffix(timestep, noisy_actions)

    assert action_time_embs.dtype == torch.float32
    assert calls == ["action_proj_func", "mlp_func"]


def test_eo1_forward_loss_stays_fp32_inside_global_autocast():
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))
    state_token_id = 9
    action_token_id = 7

    input_ids = torch.cat(
        [
            torch.full((1, 1), state_token_id, dtype=torch.long),
            torch.full((1, config.chunk_size), action_token_id, dtype=torch.long),
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(1, config.max_state_dim, dtype=torch.float32)
    action = torch.randn(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            states=states,
            action=action,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )

    assert loss is not None
    assert loss.dtype == torch.float32


def test_eo1_forward_uses_manual_checkpointing(monkeypatch):
    config = make_test_config(dtype="bfloat16")
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))
    calls = patch_checkpoint_to_record(monkeypatch)
    model.gradient_checkpointing_enable()
    model.train()

    batch_size = 1
    state_token_id = 9
    action_token_id = 7
    input_ids = torch.cat(
        [
            torch.full((batch_size, 1), state_token_id, dtype=torch.long),
            torch.full((batch_size, config.chunk_size), action_token_id, dtype=torch.long),
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(batch_size, config.max_state_dim, dtype=torch.float32)
    action = torch.randn(batch_size, config.chunk_size, config.max_action_dim, dtype=torch.float32)

    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        states=states,
        action=action,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )

    assert loss is not None
    assert calls == [
        "input_embed_func",
        "state_proj_func",
        "action_proj_func",
        "mlp_func",
        "vlm_forward_func",
        "action_out_proj_func",
    ]
    model_call = model.vlm_backbone.model_calls[-1]
    assert model_call["position_ids"] is None
    torch.testing.assert_close(model_call["mm_token_type_ids"], mm_token_type_ids)


def test_eo1_embed_prefix_uses_manual_checkpointing(monkeypatch):
    config = make_test_config(dtype="bfloat16")
    backbone = DummyVLMBackbone(config.text_config.hidden_size)
    model = EO1VisionFlowMatchingModel(config, backbone)
    calls = patch_checkpoint_to_record(monkeypatch)
    model.gradient_checkpointing_enable()
    model.train()

    input_ids = torch.tensor([[11, 12, 13]], dtype=torch.long)
    states = torch.randn(1, config.max_state_dim, dtype=torch.float32)

    inputs_embeds = model.embed_prefix(
        input_ids=input_ids,
        states=states,
        state_token_id=11,
    )

    assert inputs_embeds.shape == (1, input_ids.shape[1], config.text_config.hidden_size)
    assert calls == ["input_embed_func", "state_proj_func"]


def test_eo1_padded_actions_follow_standard_diffusion_targets():
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

    loss = model(
        input_ids=torch.cat(
            [
                torch.full((1, 1), 9, dtype=torch.long),
                torch.full((1, config.chunk_size), action_token_id, dtype=torch.long),
            ],
            dim=1,
        ),
        attention_mask=torch.ones(1, config.chunk_size + 1, dtype=torch.long),
        mm_token_type_ids=torch.zeros(1, config.chunk_size + 1, dtype=torch.int),
        states=torch.randn(1, config.max_state_dim, dtype=torch.float32),
        action=action,
        state_token_id=9,
        action_is_pad=action_is_pad,
        action_token_id=action_token_id,
    )

    assert loss is not None
    assert torch.allclose(
        captured["noisy_actions"],
        torch.full((1, config.chunk_size, config.max_action_dim), 2.0, dtype=torch.float32),
    )


def test_eo1_flow_loss_ignores_padded_action_dimensions():
    config = make_test_config(
        dtype="bfloat16",
        supervise_padding_action_dims=False,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
    )
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    class ZeroActionOutProj(nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim

        @property
        def dtype(self):
            return torch.float32

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                hidden_states.shape[0],
                self.output_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )

    model.sample_time = lambda bsize, device: torch.full((bsize,), 0.5, dtype=torch.float32, device=device)
    model.sample_noise = lambda shape, device: torch.zeros(shape, dtype=torch.float32, device=device)
    model.embed_suffix = lambda timestep, noisy_actions: torch.zeros(
        noisy_actions.shape[0],
        noisy_actions.shape[1],
        config.text_config.hidden_size,
        dtype=torch.float32,
        device=noisy_actions.device,
    )
    model.action_out_proj = ZeroActionOutProj(config.max_action_dim)

    action = torch.zeros(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)
    action[:, :, : config.output_features["action"].shape[0]] = 1.0

    loss = model(
        input_ids=torch.cat(
            [
                torch.full((1, 1), 9, dtype=torch.long),
                torch.full((1, config.chunk_size), 7, dtype=torch.long),
            ],
            dim=1,
        ),
        attention_mask=torch.ones(1, config.chunk_size + 1, dtype=torch.long),
        mm_token_type_ids=torch.zeros(1, config.chunk_size + 1, dtype=torch.int),
        states=torch.randn(1, config.max_state_dim, dtype=torch.float32),
        action=action,
        state_token_id=9,
        action_token_id=7,
    )

    assert loss is not None
    assert loss.item() == pytest.approx(1.0)


def test_eo1_flow_loss_can_supervise_padded_action_dimensions():
    config = make_test_config(
        dtype="bfloat16",
        supervise_padding_action_dims=True,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
    )
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    class ZeroActionOutProj(nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim

        @property
        def dtype(self):
            return torch.float32

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                hidden_states.shape[0],
                self.output_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )

    model.sample_time = lambda bsize, device: torch.full((bsize,), 0.5, dtype=torch.float32, device=device)
    model.sample_noise = lambda shape, device: torch.zeros(shape, dtype=torch.float32, device=device)
    model.embed_suffix = lambda timestep, noisy_actions: torch.zeros(
        noisy_actions.shape[0],
        noisy_actions.shape[1],
        config.text_config.hidden_size,
        dtype=torch.float32,
        device=noisy_actions.device,
    )
    model.action_out_proj = ZeroActionOutProj(config.max_action_dim)

    action = torch.zeros(1, config.chunk_size, config.max_action_dim, dtype=torch.float32)
    action[:, :, : config.output_features["action"].shape[0]] = 1.0

    loss = model(
        input_ids=torch.cat(
            [
                torch.full((1, 1), 9, dtype=torch.long),
                torch.full((1, config.chunk_size), 7, dtype=torch.long),
            ],
            dim=1,
        ),
        attention_mask=torch.ones(1, config.chunk_size + 1, dtype=torch.long),
        mm_token_type_ids=torch.zeros(1, config.chunk_size + 1, dtype=torch.int),
        states=torch.randn(1, config.max_state_dim, dtype=torch.float32),
        action=action,
        state_token_id=9,
        action_token_id=7,
    )

    assert loss is not None
    assert loss.item() == pytest.approx(0.25)


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
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(batch_size, config.max_state_dim, dtype=torch.float32)

    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
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
    prefill = model.vlm_backbone.model_calls[0]
    assert prefill["position_ids"] is not None
    assert prefill["position_ids"].shape == (3, batch_size, 1)
    torch.testing.assert_close(prefill["mm_token_type_ids"], mm_token_type_ids[:, :1])


def test_eo1_sample_actions_crops_cache_back_to_prefix_each_step(monkeypatch):
    config = make_test_config(dtype="bfloat16", chunk_size=4, n_action_steps=4, num_denoise_steps=3)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    class TrackingCache:
        def __init__(self, seq_length: int):
            self.seq_length = seq_length
            self.crop_calls: list[int] = []

        def get_seq_length(self) -> int:
            return self.seq_length

        def crop(self, prefix_len: int) -> None:
            self.crop_calls.append(prefix_len)
            self.seq_length = prefix_len

    denoise_cache_lengths: list[int] = []
    final_cache: TrackingCache | None = None

    def fake_model(
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: TrackingCache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        nonlocal final_cache
        assert inputs_embeds is not None
        if past_key_values is None:
            cache = TrackingCache(inputs_embeds.shape[1])
            final_cache = cache
        else:
            denoise_cache_lengths.append(past_key_values.get_seq_length())
            past_key_values.seq_length += inputs_embeds.shape[1]
            cache = past_key_values
        return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=cache)

    monkeypatch.setattr(model.vlm_backbone, "forward", fake_model)

    action_token_id = 7
    state_token_id = 9
    input_ids = torch.tensor(
        [[0, 0, state_token_id, 21, action_token_id, action_token_id, action_token_id, action_token_id]]
    )
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(1, config.max_state_dim, dtype=torch.float32)

    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        states=states,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )

    assert actions.shape == (1, config.chunk_size, config.max_action_dim)
    assert denoise_cache_lengths == [4, 4, 4]
    assert final_cache is not None
    assert final_cache.crop_calls == [4, 4, 4]


def test_eo1_sample_actions_supports_single_sample_eval_denoising():
    config = make_test_config(dtype="bfloat16", chunk_size=4, n_action_steps=4, num_denoise_steps=2)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    action_token_id = 7
    state_token_id = 9
    input_ids = torch.tensor(
        [[0, 0, state_token_id, 21, action_token_id, action_token_id, action_token_id, action_token_id]]
    )
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(1, config.max_state_dim, dtype=torch.float32)

    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        states=states,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )

    assert actions.shape == (1, config.chunk_size, config.max_action_dim)
    assert len(model.vlm_backbone.model_calls) == 1 + config.num_denoise_steps


def test_eo1_sample_actions_uses_aligned_left_padded_eval_prompts():
    config = make_test_config(dtype="bfloat16", chunk_size=4, n_action_steps=4, num_denoise_steps=2)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    action_token_id = 7
    state_token_id = 9
    input_ids = torch.tensor(
        [
            [0, 0, state_token_id, 102, 103, 7, 7, 7, 7, 201],
            [111, 112, state_token_id, 114, 115, 7, 7, 7, 7, 202],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(2, config.max_state_dim, dtype=torch.float32)

    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        states=states,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )

    assert actions.shape == (2, config.chunk_size, config.max_action_dim)
    assert len(model.vlm_backbone.model_calls) == 1 + config.num_denoise_steps

    prefill, step_0, step_1 = model.vlm_backbone.model_calls
    assert prefill["inputs_embeds"].shape == (2, 5, config.text_config.hidden_size)
    assert prefill["attention_mask"][0].tolist() == [0, 0, 1, 1, 1]
    assert prefill["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
    torch.testing.assert_close(prefill["mm_token_type_ids"], mm_token_type_ids[:, :5])
    assert prefill["position_ids"].shape == (3, 2, 5)
    assert step_0["inputs_embeds"].shape[1] == config.chunk_size
    torch.testing.assert_close(step_0["attention_mask"], attention_mask[:, :9])
    torch.testing.assert_close(step_1["attention_mask"], attention_mask[:, :9])
    assert step_0["position_ids"].shape == (3, 2, config.chunk_size)
    assert step_1["position_ids"].shape == (3, 2, config.chunk_size)


def test_eo1_sample_actions_raises_for_misaligned_action_spans():
    config = make_test_config(dtype="bfloat16", chunk_size=4, n_action_steps=4, num_denoise_steps=2)
    model = EO1VisionFlowMatchingModel(config, DummyVLMBackbone(config.text_config.hidden_size))

    action_token_id = 7
    state_token_id = 9
    input_ids = torch.tensor(
        [
            [101, 102, state_token_id, 7, 7, 7, 7, 201, 202, 0, 0, 0],
            [111, 112, state_token_id, 114, 115, 7, 7, 7, 7, 211, 212, 213],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    states = torch.randn(2, config.max_state_dim, dtype=torch.float32)

    with pytest.raises(ValueError, match="same action token mask after left padding"):
        model.sample_actions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            states=states,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )


def test_eo1_state_dict_keeps_flow_head_weights_in_fp32():
    config = make_test_config(dtype="bfloat16")
    backbone = DummyVLMBackbone(config.text_config.hidden_size).to(dtype=torch.bfloat16)
    model = EO1VisionFlowMatchingModel(config, backbone)

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
