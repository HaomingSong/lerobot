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

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

COSMOS3_LEFT_IMAGE = f"{OBS_IMAGES}.over_shoulder_left_camera"
COSMOS3_RIGHT_IMAGE = f"{OBS_IMAGES}.over_shoulder_right_camera"
COSMOS3_WRIST_IMAGE = f"{OBS_IMAGES}.wrist_cam"

COSMOS3_DROID_DOMAIN_ID = 8
COSMOS3_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
    "opposite sides, with the robot visible."
)

_TRANSFORMER_CONFIG_DROP_KEYS = {
    "_class_name",
    "_diffusers_version",
    "dtype",
    "freeze_und",
    "hidden_act",
    "initializer_range",
    "joint_attn_implementation",
    "max_action_dim",
    "max_position_embeddings",
    "model_type",
    "position_embedding_type",
    "qk_norm",
    "qk_norm_for_diffusion",
    "qk_norm_for_text",
    "temporal_compression_factor_sound",
    "use_cache",
    "use_moe",
    "video_temporal_causal",
}

_VAE_CONFIG_DROP_KEYS = {"_class_name", "_diffusers_version", "clip_output"}
_SCHEDULER_CONFIG_DROP_KEYS = {"_class_name", "_diffusers_version"}


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _without_keys(config: dict[str, Any] | None, keys: set[str]) -> dict[str, Any] | None:
    if config is None:
        return None
    cleaned = deepcopy(config)
    for key in keys:
        cleaned.pop(key, None)
    return cleaned


def _native_action_scheduler_config(config: dict[str, Any] | None, *, shift: float) -> dict[str, Any]:
    normalized = _without_keys(config, _SCHEDULER_CONFIG_DROP_KEYS) or {}
    normalized.update(
        {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
            "flow_shift": float(shift),
            "timestep_spacing": "linspace",
            "final_sigmas_type": "zero",
        }
    )
    return normalized


@PreTrainedConfig.register_subclass("cosmos3")
@dataclass
class Cosmos3Config(PreTrainedConfig):
    """Configuration for the Diffusers-model-level Cosmos3 policy integration."""

    # Model-level initialization sources. `diffusers_model_name_or_path` may point
    # at a Diffusers-style Cosmos3 checkpoint with transformer/, vae/, scheduler/,
    # and text tokenizer files. LeRobot checkpoints store the resolved sub-configs
    # below and load weights via `PreTrainedPolicy.from_pretrained`.
    diffusers_model_name_or_path: str | None = None
    base_model_name_or_path: str | None = None
    qwen3_vl_name_or_path: str | None = None
    text_processor_name_or_path: str | None = None
    transformer_config: dict[str, Any] | None = None
    wan_vae_config: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None

    # Public model-level loading controls.
    load_pretrained_weights: bool = True
    freeze_vae: bool = True
    drop_sound_modules: bool = True
    copy_understanding_to_generation_expert: bool = False
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"
    transformer_attention_backend: str | None = None
    local_files_only: bool = True

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
    eos_token_id: int = 151645
    start_of_generation_token_id: int = 151652
    mode: str = "policy"
    viewpoint: str = "concat_view"
    additional_view_description: str = COSMOS3_CONCAT_VIEW_DESCRIPTION
    conditioning_fps: float = 15.0
    resolution_tier: int = 480
    guidance_scale: float = 3.0
    num_inference_steps: int = 4
    shift: float = 5.0
    seed: int = 0
    deterministic_seed: bool = False
    generate_video: bool = False
    output_type: str = "latent"
    train_time_video_distribution: str = "waver"
    video_loss_weight: float = 10.0
    action_loss_weight: float = 10.0
    normalize_loss_by_active: bool = False

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
        self._populate_serialized_subconfigs()
        self.scheduler_config = _native_action_scheduler_config(self.scheduler_config, shift=self.shift)
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

    def _populate_serialized_subconfigs(self) -> None:
        source = self.diffusers_model_name_or_path or self.base_model_name_or_path
        if source is None:
            return

        model_path = Path(source)
        if not model_path.is_dir():
            return

        if self.transformer_config is None:
            self.transformer_config = _read_json_if_present(model_path / "transformer" / "config.json")
        if self.wan_vae_config is None:
            self.wan_vae_config = _read_json_if_present(model_path / "vae" / "config.json")
        if self.scheduler_config is None:
            self.scheduler_config = _read_json_if_present(model_path / "scheduler" / "scheduler_config.json")
        if self.text_processor_name_or_path is None:
            text_tokenizer_path = model_path / "text_tokenizer"
            self.text_processor_name_or_path = str(
                text_tokenizer_path if text_tokenizer_path.is_dir() else model_path
            )

    @property
    def transformer_backbone_config(self) -> dict[str, Any]:
        config = _without_keys(self.transformer_config, _TRANSFORMER_CONFIG_DROP_KEYS)
        if config is None:
            config = {
                "action_dim": self.max_action_dim,
                "action_gen": True,
                "sound_dim": None,
                "sound_gen": False,
            }
        if self.drop_sound_modules:
            config["sound_dim"] = None
            config["sound_gen"] = False
        config.setdefault("action_dim", self.max_action_dim)
        config.setdefault("action_gen", True)
        return config

    @property
    def vae_config(self) -> dict[str, Any] | None:
        return _without_keys(self.wan_vae_config, _VAE_CONFIG_DROP_KEYS)

    @property
    def unipc_scheduler_config(self) -> dict[str, Any] | None:
        return _native_action_scheduler_config(self.scheduler_config, shift=self.shift)

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
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size + 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
