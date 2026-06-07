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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_cosmos3 import Cosmos3Config

COSMOS3_VIDEO = "cosmos3_video"
COSMOS3_COMPOSED_IMAGE = "cosmos3_composed_image"
COSMOS3_ACTION_CONDITION = "cosmos3_action_condition"
COSMOS3_ACTION_CONDITION_MASK = "cosmos3_action_condition_mask"
COSMOS3_ACTION_DOMAIN_ID = "cosmos3_action_domain_id"
COSMOS3_CONDITIONING_FPS = "cosmos3_conditioning_fps"
COSMOS3_RAW_ACTION_DIM = "cosmos3_raw_action_dim"
COSMOS3_IMAGE_SIZE = "cosmos3_image_size"
COSMOS3_PROMPT = "cosmos3_prompt"
COSMOS3_SEQUENCE_METADATA = "cosmos3_sequence_metadata"


def _as_batched_image_tensor(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeezed = True
    elif image.ndim == 4:
        squeezed = False
    else:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(image.shape)}")

    if image.shape[-1] in {1, 3, 4}:
        image = image.permute(0, 3, 1, 2)
    elif image.shape[1] not in {1, 3, 4}:
        raise ValueError(f"Could not infer image channel dimension from shape={tuple(image.shape)}")
    if image.shape[1] == 4:
        image = image[:, :3]
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    return image.contiguous(), squeezed


def _to_uint8_nhwc(image: torch.Tensor) -> np.ndarray:
    image, _ = _as_batched_image_tensor(image)
    if image.dtype.is_floating_point:
        image = image.clamp(0.0, 1.0).mul(255.0)
    image = image.round().clamp(0, 255).to(torch.uint8)
    return image.permute(0, 2, 3, 1).cpu().numpy()


def resize_with_pad_uint8(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match RoboLab's openpi-client resize_with_pad behavior for NHWC uint8 images."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int) -> np.ndarray:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


def compose_robolab_cosmos3_image(
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    wrist_image: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compose RoboLab's wrist/top and shoulder/bottom image layout as uint8 NHWC."""
    left = resize_with_pad_uint8(_to_uint8_nhwc(left_image), image_height, image_width)
    right = resize_with_pad_uint8(_to_uint8_nhwc(right_image), image_height, image_width)
    wrist = resize_with_pad_uint8(_to_uint8_nhwc(wrist_image), image_height, image_width)

    bottom_size = (image_height // 2, image_width // 2)
    left_half = torch.from_numpy(left).permute(0, 3, 1, 2).float()
    left_half = F.interpolate(left_half, size=bottom_size, mode="bilinear")
    left_half = left_half.permute(0, 2, 3, 1).numpy().astype(wrist.dtype)

    right_half = torch.from_numpy(right).permute(0, 3, 1, 2).float()
    right_half = F.interpolate(right_half, size=bottom_size, mode="bilinear")
    right_half = right_half.permute(0, 2, 3, 1).numpy().astype(wrist.dtype)

    bottom = np.concatenate((left_half, right_half), axis=2)
    composed = np.concatenate((wrist, bottom), axis=1)
    return torch.from_numpy(composed)


def _normalise_prompt_list(prompts: Any, batch_size: int) -> list[str]:
    if isinstance(prompts, str):
        return [prompts] * batch_size
    if isinstance(prompts, tuple):
        prompts = list(prompts)
    if isinstance(prompts, list) and len(prompts) == batch_size and all(isinstance(p, str) for p in prompts):
        return prompts
    raise ValueError(f"Expected a prompt string or list[str] with batch_size={batch_size}, got {prompts!r}")


@dataclass
@ProcessorStepRegistry.register(name="cosmos3_robolab_pack_inputs")
class Cosmos3RoboLabPackInputsStep(ComplementaryDataProcessorStep):
    left_image_key: str
    right_image_key: str
    wrist_image_key: str
    image_height: int
    image_width: int
    chunk_size: int
    raw_action_dim: int
    max_action_dim: int
    use_state: bool
    history_length: int
    invert_gripper: bool
    domain_id: int
    conditioning_fps: float
    resolution_tier: int
    prompt_key: str = "task"
    composed_image_height: int = 540
    composed_image_width: int = 640

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for Cosmos3RoboLabPackInputsStep.")

        left = observation.get(self.left_image_key)
        right = observation.get(self.right_image_key)
        wrist = observation.get(self.wrist_image_key)
        if left is None or right is None or wrist is None:
            raise ValueError(
                "Cosmos3 requires left, right, and wrist image observations. "
                f"Missing keys from {[self.left_image_key, self.right_image_key, self.wrist_image_key]}."
            )

        composed = compose_robolab_cosmos3_image(
            left,
            right,
            wrist,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        if composed.shape[1:3] != (self.composed_image_height, self.composed_image_width):
            raise ValueError(
                "Unexpected composed Cosmos3 image shape: "
                f"{tuple(composed.shape[1:3])}, expected {(self.composed_image_height, self.composed_image_width)}."
            )

        batch_size = composed.shape[0]
        video = torch.zeros(
            batch_size,
            3,
            self.chunk_size + 1,
            self.composed_image_height,
            self.composed_image_width,
            dtype=torch.uint8,
        )
        video[:, :, 0] = composed.permute(0, 3, 1, 2)

        action_len = self.chunk_size + int(self.use_state)
        action_condition = torch.zeros(batch_size, action_len, self.raw_action_dim, dtype=torch.float32)
        action_condition_mask = torch.zeros(batch_size, action_len, 1, dtype=torch.float32)
        if self.use_state:
            state = observation.get(OBS_STATE)
            if state is None:
                raise ValueError(f"{OBS_STATE} is required when Cosmos3 use_state=True.")
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(dtype=torch.float32)
            if state.shape[0] != batch_size:
                raise ValueError("Batch size mismatch between Cosmos3 images and state.")
            if state.shape[-1] < self.raw_action_dim:
                raise ValueError(
                    f"Cosmos3 state width must be at least raw_action_dim={self.raw_action_dim}, "
                    f"got {state.shape[-1]}."
                )
            state = state[:, : self.raw_action_dim].clone()
            if self.invert_gripper:
                state[:, -1] = 1.0 - state[:, -1]
            action_condition[:, 0] = state
            action_condition_mask[:, 0, 0] = 1.0

        prompts = _normalise_prompt_list(complementary_data.get(self.prompt_key), batch_size)
        complementary_data[COSMOS3_PROMPT] = prompts
        complementary_data[COSMOS3_COMPOSED_IMAGE] = composed
        complementary_data[COSMOS3_VIDEO] = video
        complementary_data[COSMOS3_ACTION_CONDITION] = action_condition
        complementary_data[COSMOS3_ACTION_CONDITION_MASK] = action_condition_mask
        complementary_data[COSMOS3_ACTION_DOMAIN_ID] = torch.full(
            (batch_size,), self.domain_id, dtype=torch.long
        )
        complementary_data[COSMOS3_CONDITIONING_FPS] = torch.full(
            (batch_size,), self.conditioning_fps, dtype=torch.float32
        )
        complementary_data[COSMOS3_RAW_ACTION_DIM] = torch.full(
            (batch_size,), self.raw_action_dim, dtype=torch.long
        )
        complementary_data[COSMOS3_IMAGE_SIZE] = torch.tensor(
            [self.composed_image_height, self.composed_image_width],
            dtype=torch.long,
        ).expand(batch_size, -1)
        complementary_data[COSMOS3_SEQUENCE_METADATA] = {
            "resolution_tier": self.resolution_tier,
            "history_length": self.history_length,
            "action_start_frame_offset": 0 if self.use_state else 1,
        }
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "left_image_key": self.left_image_key,
            "right_image_key": self.right_image_key,
            "wrist_image_key": self.wrist_image_key,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "chunk_size": self.chunk_size,
            "raw_action_dim": self.raw_action_dim,
            "max_action_dim": self.max_action_dim,
            "use_state": self.use_state,
            "history_length": self.history_length,
            "invert_gripper": self.invert_gripper,
            "domain_id": self.domain_id,
            "conditioning_fps": self.conditioning_fps,
            "resolution_tier": self.resolution_tier,
            "prompt_key": self.prompt_key,
            "composed_image_height": self.composed_image_height,
            "composed_image_width": self.composed_image_width,
        }


def make_cosmos3_pre_post_processors(
    config: Cosmos3Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    config.validate_features()

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Cosmos3RoboLabPackInputsStep(
            left_image_key=config.left_image_key,
            right_image_key=config.right_image_key,
            wrist_image_key=config.wrist_image_key,
            image_height=config.image_height,
            image_width=config.image_width,
            composed_image_height=config.composed_image_height,
            composed_image_width=config.composed_image_width,
            chunk_size=config.chunk_size,
            raw_action_dim=config.raw_action_dim,
            max_action_dim=config.max_action_dim,
            use_state=config.use_state,
            history_length=config.history_length,
            invert_gripper=config.invert_gripper,
            domain_id=config.domain_id,
            conditioning_fps=config.conditioning_fps,
            resolution_tier=config.resolution_tier,
            prompt_key=config.prompt_key,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
