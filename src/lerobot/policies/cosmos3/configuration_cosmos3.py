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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

COSMOS3_LEFT_IMAGE = f"{OBS_IMAGES}.over_shoulder_left_camera"
COSMOS3_RIGHT_IMAGE = f"{OBS_IMAGES}.over_shoulder_right_camera"
COSMOS3_WRIST_IMAGE = f"{OBS_IMAGES}.wrist_cam"

COSMOS3_DROID_DOMAIN_ID = 8
COSMOS3_CONCAT_VIEW_DESCRIPTION = (
    "This video contains concatenated views from multiple camera perspectives. "
    "The top half is a wrist-mounted camera view. The bottom half contains "
    "left and right third-person shoulder camera views."
)


@PreTrainedConfig.register_subclass("cosmos3")
@dataclass
class Cosmos3Config(PreTrainedConfig):
    """Configuration for Cosmos3 action-policy integration in LeRobot."""

    # Optional original Diffusers/Cosmos checkpoint used to initialize the runtime pipeline.
    diffusers_model_name_or_path: str | None = None
    qwen3_vl_config: dict | None = None
    wan_vae_config: dict | None = None
    transformer_config: dict | None = None

    # Loading controls. Unit tests and checkpoint conversion can instantiate the policy
    # without loading the large Diffusers pipeline by setting this to False.
    load_diffusers_pipeline: bool = True
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"
    attn_implementation: str | None = None
    enable_safety_checker: bool = False

    # RoboLab/DROID policy contract.
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    raw_action_dim: int = 8
    max_action_dim: int = 64
    joint_position_dim: int = 7
    gripper_position_dim: int = 1
    use_state: bool = True
    history_length: int = 1
    action_space: str = "joint_pos"
    invert_gripper: bool = True

    # Cosmos3 action generation settings matching the RoboLab policy server defaults.
    domain_name: str = "droid_lerobot"
    domain_id: int = COSMOS3_DROID_DOMAIN_ID
    mode: str = "policy"
    viewpoint: str = "concat_view"
    additional_view_description: str = COSMOS3_CONCAT_VIEW_DESCRIPTION
    conditioning_fps: float = 15.0
    resolution_tier: int = 480
    guidance_scale: float = 3.0
    num_inference_steps: int = 4
    shift: float = 5.0
    seed: int = 42
    deterministic_seed: bool = True
    generate_video: bool = False
    output_type: str = "latent"

    # Three-view image composition mirrors RoboLab/policies/cosmos3/client.py.
    left_image_key: str = COSMOS3_LEFT_IMAGE
    right_image_key: str = COSMOS3_RIGHT_IMAGE
    wrist_image_key: str = COSMOS3_WRIST_IMAGE
    image_height: int = 360
    image_width: int = 640
    composed_image_height: int = 540
    composed_image_width: int = 640
    prompt_key: str = "task"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Conservative training defaults until task-specific recipes are added.
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Invalid dtype: {self.dtype!r}")
        if self.mode != "policy":
            raise ValueError("Cosmos3Config currently supports only action mode='policy'.")
        if self.action_space != "joint_pos":
            raise ValueError("Cosmos3Config currently supports only action_space='joint_pos'.")
        if self.history_length < int(self.use_state):
            raise ValueError("history_length must be at least 1 when use_state=True.")
        if self.raw_action_dim != self.joint_position_dim + self.gripper_position_dim:
            raise ValueError("raw_action_dim must equal joint_position_dim + gripper_position_dim.")

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        default_image_shape = (3, self.image_height, self.image_width)
        for image_key in (self.left_image_key, self.right_image_key, self.wrist_image_key):
            self.input_features.setdefault(
                image_key,
                PolicyFeature(type=FeatureType.VISUAL, shape=default_image_shape),
            )

        self.input_features.setdefault(
            OBS_STATE,
            PolicyFeature(type=FeatureType.STATE, shape=(self.raw_action_dim,)),
        )
        self.output_features.setdefault(
            ACTION,
            PolicyFeature(type=FeatureType.ACTION, shape=(self.raw_action_dim,)),
        )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
