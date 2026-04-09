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

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.eo1 import processor_eo1 as processor_module
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.processor.converters import create_transition


def build_input_features():
    return {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        processor_module.OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }


def build_output_features():
    return {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }


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
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2, dtype=torch.float32),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


class FakeQwenProcessorFactory:
    fake_processor: FakeQwenProcessor | None = None

    @classmethod
    def from_pretrained(cls, processor_name):
        cls.fake_processor = FakeQwenProcessor()
        return cls.fake_processor


def build_messages():
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
        ]
    ]


def test_eo1_conversation_template_restores_uint8_images():
    step = processor_module.EO1ConversationTemplateStep(
        input_features=build_input_features(),
        chunk_size=4,
    )
    raw_image = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)

    transition = create_transition(
        observation={"observation.image": raw_image.float() / 255.0},
        complementary_data={"task": ["pick", "place"]},
    )

    output = step(transition)
    messages = output[processor_module.TransitionKey.COMPLEMENTARY_DATA]["messages"]
    restored = messages[0][1]["content"][0]["image"]

    assert restored.dtype == torch.uint8
    assert torch.equal(restored, raw_image[0])
    assert messages[0][1]["content"][1]["text"].endswith("pick<|vla|>")
    assert messages[1][1]["content"][1]["text"].endswith("place<|vla|>")


def test_eo1_qwen_processor_uses_right_padding_for_supervised_batches(monkeypatch):
    monkeypatch.setattr(processor_module, "Qwen2_5_VLProcessor", FakeQwenProcessorFactory)
    step = processor_module.EO1QwenProcessorStep(
        processor_name="dummy",
        image_min_pixels=3136,
        image_max_pixels=12845056,
    )

    transition = create_transition(
        action=torch.zeros(1, dtype=torch.float32),
        complementary_data={"messages": build_messages()},
    )

    output = step(transition)
    fake_processor = FakeQwenProcessorFactory.fake_processor
    assert fake_processor is not None
    kwargs = fake_processor.calls[-1]["kwargs"]

    assert kwargs["padding_side"] == "right"
    assert kwargs["min_pixels"] == 3136
    assert kwargs["max_pixels"] == 12845056
    assert "do_resize" not in kwargs
    assert output[processor_module.TransitionKey.COMPLEMENTARY_DATA]["state_token_id"] == 11
    assert output[processor_module.TransitionKey.COMPLEMENTARY_DATA]["action_token_id"] == 12


def test_eo1_qwen_processor_uses_left_padding_for_rollout_batches(monkeypatch):
    monkeypatch.setattr(processor_module, "Qwen2_5_VLProcessor", FakeQwenProcessorFactory)
    step = processor_module.EO1QwenProcessorStep(
        processor_name="dummy",
        image_min_pixels=3136,
        image_max_pixels=12845056,
    )

    transition = create_transition(
        action=None,
        complementary_data={"messages": build_messages()},
    )

    step(transition)
    fake_processor = FakeQwenProcessorFactory.fake_processor
    assert fake_processor is not None

    assert fake_processor.calls[-1]["kwargs"]["padding_side"] == "left"


def test_make_eo1_pre_post_processors_keeps_visual_feature_shapes(monkeypatch):
    monkeypatch.setattr(processor_module, "Qwen2_5_VLProcessor", FakeQwenProcessorFactory)
    config = EO1Config(
        vlm_base="dummy",
        vlm_config={},
        device="cpu",
        input_features=build_input_features(),
        output_features=build_output_features(),
    )

    preprocessor, _ = processor_module.make_eo1_pre_post_processors(config=config, dataset_stats=None)

    transformed = preprocessor.transform_features(
        {
            PipelineFeatureType.OBSERVATION: dict(config.input_features),
            PipelineFeatureType.ACTION: dict(config.output_features),
        }
    )

    assert transformed[PipelineFeatureType.OBSERVATION]["observation.image"].shape == (3, 16, 16)
    assert "EO1ConversationTemplateStep" in [type(step).__name__ for step in preprocessor.steps]
    assert "EO1RestoreRawUint8Step" not in [type(step).__name__ for step in preprocessor.steps]
    assert "EO1ImageSmartResizeStep" not in [type(step).__name__ for step in preprocessor.steps]
