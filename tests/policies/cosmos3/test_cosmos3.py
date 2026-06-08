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
from lerobot.policies.cosmos3.modeling_cosmos3 import (
    Cosmos3Policy,
    _prepare_native_action_video_conditioning,
)
from lerobot.policies.cosmos3.processor_cosmos3 import (
    COSMOS3_ACTION_CONDITION,
    COSMOS3_ACTION_CONDITION_MASK,
    COSMOS3_CLEAN_ACTION,
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
        transformer_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "patch_latent_dim": 4,
            "latent_channel": 4,
            "latent_patch_size": 1,
            "action_dim": 64,
            "action_gen": True,
            "vocab_size": 128,
            "rope_scaling": {"mrope_section": [2, 1, 1]},
        },
        wan_vae_config={
            "base_dim": 4,
            "decoder_base_dim": 4,
            "z_dim": 4,
            "dim_mult": [1],
            "num_res_blocks": 1,
            "attn_scales": [],
            "temperal_downsample": [],  # spellchecker:disable-line
            "scale_factor_temporal": 1,
            "scale_factor_spatial": 1,
        },
        scheduler_config={
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "use_karras_sigmas": False,
            "flow_shift": 5.0,
        },
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


def constant_image_sequence(value: int, num_frames: int) -> torch.Tensor:
    return torch.full((1, num_frames, 3, 360, 640), value / 255.0, dtype=torch.float32)


def test_cosmos3_factory_registration():
    cfg = make_policy_config("cosmos3", device="cpu")

    assert isinstance(cfg, Cosmos3Config)
    assert get_policy_class("cosmos3") is Cosmos3Policy

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_cosmos3_config_ignores_external_checkpoint_layout(tmp_path):
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text('{"hidden_size": 999}', encoding="utf-8")

    cfg = Cosmos3Config(
        device="cpu",
        diffusers_model_name_or_path=str(tmp_path),
        transformer_config={"hidden_size": 8, "action_dim": 64, "action_gen": True},
        wan_vae_config={"z_dim": 4},
        scheduler_config={"flow_shift": 5.0},
    )

    assert cfg.transformer_config == {"hidden_size": 8, "action_dim": 64, "action_gen": True}
    assert cfg.text_processor_name_or_path is None


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


def test_cosmos3_robolab_processor_packs_training_video_and_clean_actions():
    cfg = make_config()
    preprocessor, _ = make_cosmos3_pre_post_processors(cfg)
    state = torch.zeros(1, cfg.chunk_size + 1, 8, dtype=torch.float32)
    state[:, :, -1] = 0.25
    actions = torch.zeros(1, cfg.chunk_size, 8, dtype=torch.float32)
    actions[:, :, -1] = 0.75

    batch = {
        COSMOS3_LEFT_IMAGE: constant_image_sequence(10, cfg.chunk_size + 1),
        COSMOS3_RIGHT_IMAGE: constant_image_sequence(20, cfg.chunk_size + 1),
        COSMOS3_WRIST_IMAGE: constant_image_sequence(30, cfg.chunk_size + 1),
        OBS_STATE: state,
        ACTION: actions,
        "task": ["Pick up the banana and place it in the bowl."],
    }

    processed = preprocessor(batch)

    assert processed[COSMOS3_COMPOSED_IMAGE].shape == (1, cfg.chunk_size + 1, 540, 640, 3)
    assert processed[COSMOS3_VIDEO].shape == (1, 3, cfg.chunk_size + 1, 540, 640)
    torch.testing.assert_close(
        processed[COSMOS3_VIDEO][:, :, 5],
        processed[COSMOS3_COMPOSED_IMAGE][:, 5].permute(0, 3, 1, 2),
    )

    clean_action = processed[COSMOS3_CLEAN_ACTION]
    assert clean_action.shape == (1, cfg.chunk_size + 1, cfg.max_action_dim)
    torch.testing.assert_close(clean_action[0, 0, :8], torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.75]).float())
    torch.testing.assert_close(clean_action[0, 1, :8], torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.25]).float())
    assert torch.count_nonzero(clean_action[..., 8:]) == 0


def test_cosmos3_native_action_prompt_matches_robolab_string_transform():
    policy = Cosmos3Policy(make_config())

    prompt = policy.model._format_native_action_prompt(
        "Pick up the banana and place it in the bowl.",
        num_frames=33,
        height=544,
        width=736,
        fps=15.0,
    )

    assert prompt == (
        "Pick up the banana and place it in the bowl. "
        "This video contains concatenated views from multiple camera perspectives. "
        "The top row is from the wrist-mounted camera. "
        "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
        "opposite sides, with the robot visible. "
        "The video is 2.0 seconds long and is of 15 FPS. "
        "This video is of 544x736 resolution."
    )


def test_cosmos3_video_conditioning_matches_native_resize_contract():
    video = torch.zeros(3, 33, 540, 640, dtype=torch.uint8)
    video[:, 0] = torch.tensor([10, 20, 30], dtype=torch.uint8).view(3, 1, 1)

    frames, image_size, height, width = _prepare_native_action_video_conditioning(
        video,
        resolution_tier=480,
        num_frames=33,
        device="cpu",
        dtype=torch.float32,
    )

    assert frames.shape == (1, 3, 33, 544, 736)
    torch.testing.assert_close(image_size, torch.tensor([544.0, 736.0, 540.0, 640.0]))
    assert (height, width) == (544, 736)
    torch.testing.assert_close(frames[0, :, 0, 0, 0], torch.tensor([10, 20, 30]) / 127.5 - 1.0)


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


def test_cosmos3_masked_flow_matching_mse_matches_native_denominator():
    policy = Cosmos3Policy(make_config())
    pred = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    target = torch.tensor([[5.0, 5.0], [1.0, 1.0]])
    noisy_mask = torch.tensor([[0.0], [1.0]])

    torch.testing.assert_close(
        policy.model._masked_flow_matching_mse(pred, target, noisy_mask),
        torch.tensor(2.5),
    )

    policy.config.normalize_loss_by_active = True
    torch.testing.assert_close(
        policy.model._masked_flow_matching_mse(pred, target, noisy_mask),
        torch.tensor(5.0),
    )
