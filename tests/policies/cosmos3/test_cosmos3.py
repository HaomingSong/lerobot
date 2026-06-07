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

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.cosmos3.configuration_cosmos3 import (
    COSMOS3_LEFT_IMAGE,
    COSMOS3_RIGHT_IMAGE,
    COSMOS3_WRIST_IMAGE,
    Cosmos3Config,
)
from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3Policy
from lerobot.policies.cosmos3.processor_cosmos3 import (
    COSMOS3_ACTION_CONDITION,
    COSMOS3_ACTION_CONDITION_MASK,
    COSMOS3_COMPOSED_IMAGE,
    COSMOS3_PROMPT,
    COSMOS3_VIDEO,
    make_cosmos3_pre_post_processors,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def make_config() -> Cosmos3Config:
    return Cosmos3Config(
        device="cpu",
        load_diffusers_pipeline=False,
        input_features={
            COSMOS3_LEFT_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
            COSMOS3_RIGHT_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
            COSMOS3_WRIST_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )


def constant_image(value: int) -> torch.Tensor:
    return torch.full((3, 360, 640), value / 255.0, dtype=torch.float32)


def test_cosmos3_factory_registration():
    cfg = make_policy_config("cosmos3", device="cpu", load_diffusers_pipeline=False)

    assert isinstance(cfg, Cosmos3Config)
    assert get_policy_class("cosmos3") is Cosmos3Policy

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_cosmos3_robolab_processor_packs_native_contract():
    cfg = make_config()
    preprocessor, _ = make_cosmos3_pre_post_processors(cfg)
    state = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25], dtype=torch.float32)

    batch = {
        COSMOS3_LEFT_IMAGE: constant_image(10),
        COSMOS3_RIGHT_IMAGE: constant_image(20),
        COSMOS3_WRIST_IMAGE: constant_image(30),
        OBS_STATE: state,
        "task": "Pick up the banana and place it in the bowl.",
    }

    processed = preprocessor(batch)
    composed = processed[COSMOS3_COMPOSED_IMAGE]
    video = processed[COSMOS3_VIDEO]
    action_condition = processed[COSMOS3_ACTION_CONDITION]
    action_condition_mask = processed[COSMOS3_ACTION_CONDITION_MASK]

    assert composed.shape == (1, 540, 640, 3)
    assert torch.all(composed[:, :360] == 30)
    assert torch.all(composed[:, 360:, :320] == 10)
    assert torch.all(composed[:, 360:, 320:] == 20)

    assert video.shape == (1, 3, 33, 540, 640)
    torch.testing.assert_close(video[:, :, 0], composed.permute(0, 3, 1, 2))
    assert torch.count_nonzero(video[:, :, 1:]) == 0

    assert action_condition.shape == (1, 33, 8)
    expected_state = state.clone()
    expected_state[-1] = 1.0 - expected_state[-1]
    torch.testing.assert_close(action_condition[0, 0], expected_state)
    assert torch.count_nonzero(action_condition[0, 1:]) == 0
    torch.testing.assert_close(action_condition_mask[0, 0], torch.ones(1))
    assert torch.count_nonzero(action_condition_mask[0, 1:]) == 0
    assert processed[COSMOS3_PROMPT] == ["Pick up the banana and place it in the bowl."]


def test_cosmos3_select_action_uses_chunk_queue(monkeypatch):
    policy = Cosmos3Policy(make_config())
    fixed_chunk = torch.arange(32 * 8, dtype=torch.float32).view(1, 32, 8)
    sample_calls = {"count": 0}

    def fake_sample_actions(batch, **kwargs):
        sample_calls["count"] += 1
        return fixed_chunk

    monkeypatch.setattr(policy.model, "sample_actions", fake_sample_actions)

    action_0 = policy.select_action({})
    action_1 = policy.select_action({})

    torch.testing.assert_close(action_0, fixed_chunk[:, 0])
    torch.testing.assert_close(action_1, fixed_chunk[:, 1])
    assert sample_calls["count"] == 1
