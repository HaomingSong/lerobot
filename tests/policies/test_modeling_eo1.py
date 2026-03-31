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
from lerobot.policies.eo1.modeling_eo1 import EO1VisionFlowMatchingModel

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
