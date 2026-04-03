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

import torch

from lerobot.policies.eo1 import processor_eo1 as processor_module
from lerobot.processor.converters import create_transition


class FakeTokenizer:
    def __init__(self):
        self._token_to_id = {
            processor_module.DEFAULT_STATE_TOKEN: 11,
            processor_module.DEFAULT_ACTION_TOKEN: 12,
        }

    def add_tokens(self, tokens, special_tokens=True):
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        return self._token_to_id[token]


class FakeQwenProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.padding_sides: list[str] = []

    def apply_chat_template(self, messages, **kwargs):
        self.padding_sides.append(kwargs["padding_side"])
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2, dtype=torch.float32),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


class FakeQwenProcessorFactory:
    fake_processor = FakeQwenProcessor()

    @classmethod
    def from_pretrained(cls, processor_name):
        return cls.fake_processor


def build_messages():
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
        ]
    ]


def test_eo1_qwen_processor_uses_right_padding_for_supervised_batches(monkeypatch):
    monkeypatch.setattr(processor_module, "Qwen2_5_VLProcessor", FakeQwenProcessorFactory)
    step = processor_module.EO1QwenProcessorStep(processor_name="dummy")

    transition = create_transition(
        action=torch.zeros(1, dtype=torch.float32),
        complementary_data={"messages": build_messages()},
    )

    output = step(transition)

    assert FakeQwenProcessorFactory.fake_processor.padding_sides[-1] == "right"
    assert output[processor_module.TransitionKey.COMPLEMENTARY_DATA]["state_token_id"] == 11
    assert output[processor_module.TransitionKey.COMPLEMENTARY_DATA]["action_token_id"] == 12


def test_eo1_qwen_processor_uses_left_padding_for_rollout_batches(monkeypatch):
    monkeypatch.setattr(processor_module, "Qwen2_5_VLProcessor", FakeQwenProcessorFactory)
    step = processor_module.EO1QwenProcessorStep(processor_name="dummy")

    transition = create_transition(
        action=None,
        complementary_data={"messages": build_messages()},
    )

    step(transition)

    assert FakeQwenProcessorFactory.fake_processor.padding_sides[-1] == "left"
