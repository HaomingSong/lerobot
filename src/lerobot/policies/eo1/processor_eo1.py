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

from dataclasses import dataclass, field
from typing import Any

import torch
import torchvision.transforms.functional as F  # noqa: N812
from qwen_vl_utils.vision_process import IMAGE_FACTOR, process_vision_info, smart_resize
from torchvision.transforms import InterpolationMode

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.eo1.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

SYSTEM_MESSAGE = "You are a helpful physical assistant."

# EO-1 special tokens
ACTION_START_TOKEN = "<|action_start|>"
DEFAULT_ACTION_TOKEN = "<|action_pad|>"
ACTION_END_TOKEN = "<|action_end|>"
STATE_START_TOKEN = "<|state_start|>"
DEFAULT_STATE_TOKEN = "<|state_pad|>"
STATE_END_TOKEN = "<|state_end|>"
TASK_VLA_TOKEN = "<|vla|>"


def get_image_info(image_path, min_pixel, max_pixel, width, height):
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]

    image_input, _ = process_vision_info(messages)
    return image_input[0]


def pad_vector(vector, new_dim):
    """Can be (b s e) or (b e)"""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


@dataclass
@ProcessorStepRegistry.register(name="eo1_action_padding_processor")
class EO1ActionPaddingProcessorStep(PolicyActionProcessorStep):
    max_action_dim: int

    def __call__(self, transition):
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None or not isinstance(action, torch.Tensor):
            return new_transition

        new_transition[TransitionKey.ACTION] = self.action(action)
        return new_transition

    def action(self, action):
        return self._process_action(action)

    def _process_action(self, action):
        """Pad the action to the max_action_dim."""
        processed_action = action.clone()
        processed_action = pad_vector(processed_action, self.max_action_dim)

        return processed_action

    def get_config(self) -> dict[str, Any]:
        return {"max_action_dim": self.max_action_dim}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pad the action to the max_action_dim.
        Args:
            features: The input feature dictionary.

        Returns:
            The feature dictionary with the action padded to the max_action_dim.
        """
        action_feature = features[PipelineFeatureType.ACTION].get(ACTION)
        if action_feature:
            shape = list(action_feature.shape)
            shape[-1] = self.max_action_dim
            pad_shape = tuple(shape)
            features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=pad_shape
            )
        return features


@dataclass
@ProcessorStepRegistry.register(name="eo1_state_padding_processor")
class EO1StatePaddingProcessorStep(ObservationProcessorStep):
    max_state_dim: int

    def observation(self, observation):
        return self._process_observation(observation)

    def _process_observation(self, observation):
        """Pad the state to the max_state_dim."""

        processed_obs = observation.copy()
        if OBS_STATE not in processed_obs:
            return processed_obs
        states = processed_obs.pop(OBS_STATE)
        states = pad_vector(states, self.max_state_dim)
        processed_obs[OBS_STATE] = states
        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pad the state to the max_state_dim.
        Args:
            features: The input feature dictionary.

        Returns:
            The feature dictionary with the state padded to the max_state_dim.
        """
        state_feature = features[PipelineFeatureType.OBSERVATION].get(OBS_STATE)
        if state_feature:
            shape = list(state_feature.shape)
            shape[-1] = self.max_state_dim
            pad_shape = tuple(shape)
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=pad_shape
            )
        return features


@dataclass
@ProcessorStepRegistry.register(name="eo1_image_smart_resize_processor")
class EO1ImageSmartResizeStep(ObservationProcessorStep):
    input_features: dict[str, PolicyFeature] | dict[str, dict[str, Any]]

    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    image_resized_width: int | None = None
    image_resized_height: int | None = None
    size_factor: int = IMAGE_FACTOR

    _resized_features: dict[str, PolicyFeature] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Robust JSON deserialization handling (guard empty maps).
        if self.input_features:
            first_val = next(iter(self.input_features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.input_features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.input_features = reconstructed

        for key, value in self.input_features.items():
            if value.type != FeatureType.VISUAL:
                continue

            channels, height, width = value.shape
            if self.image_resized_width is not None and self.image_resized_height is not None:
                resized_height, resized_width = smart_resize(
                    self.image_resized_height,
                    self.image_resized_width,
                    factor=self.size_factor,
                )
            else:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.size_factor,
                    min_pixels=self.image_min_pixels,
                    max_pixels=self.image_max_pixels,
                )
            self._resized_features[key] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(channels, resized_height, resized_width)
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "input_features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.input_features.items()
            },
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "image_resized_width": self.image_resized_width,
            "image_resized_height": self.image_resized_height,
            "size_factor": self.size_factor,
        }

    def resize_image(self, image, resized_height, resized_width):
        return F.resize(
            image,
            (resized_height, resized_width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def observation(self, observation):
        return self._process_observation(observation)

    def _process_observation(self, observation):
        """Resize the image to the resized_features."""
        processed_obs = observation.copy()

        for key, value in self._resized_features.items():
            if key not in processed_obs:
                raise ValueError(f"Missing visual observation '{key}' required by {self.__class__.__name__}.")
            _, resized_height, resized_width = value.shape
            image = processed_obs.pop(key)
            image = self.resize_image(image, resized_height, resized_width)
            processed_obs[key] = image
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Resize the image to the resized_features.
        Args:
            features: The input feature dictionary.

        Returns:
            The feature dictionary with the image resized to the resized_features.
        """
        for key, value in self._resized_features.items():
            features[PipelineFeatureType.OBSERVATION][key] = value
        return features


@dataclass
@ProcessorStepRegistry.register(name="eo1_conversation_template_processor")
class EO1ConversationTemplateStep(ComplementaryDataProcessorStep):
    input_features: dict[str, PolicyFeature] | dict[str, dict[str, Any]]
    chunk_size: int

    _image_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Robust JSON deserialization handling (guard empty maps).
        if self.input_features:
            first_val = next(iter(self.input_features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.input_features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.input_features = reconstructed

        self._image_keys = [
            key for key, value in self.input_features.items() if value.type == FeatureType.VISUAL
        ]

    def complementary_data(self, complementary_data):
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Task is required for EO1ConversationTemplateStep.")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for EO1ConversationTemplateStep.")

        if OBS_STATE in observation and observation[OBS_STATE].shape[0] != len(tasks):
            raise ValueError("Batch size mismatch between observation.state and task list.")

        messages = []
        for i in range(len(tasks)):
            content = [
                *[{"type": "image", "image": observation[key][i]} for key in self._image_keys],
                {
                    "type": "text",
                    "text": (
                        f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}{tasks[i]}{TASK_VLA_TOKEN}"
                    ),
                },
            ]
            messages.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN * self.chunk_size}{ACTION_END_TOKEN}",
                            }
                        ],
                    },
                ]
            )

        complementary_data["messages"] = messages

        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step only materializes EO1-specific message objects in complementary_data.
        PipelineFeatureType tracks only ACTION and OBSERVATION, so there is no static
        feature contract change to record here.
        """
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "input_features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.input_features.items()
            },
            "chunk_size": self.chunk_size,
        }


@dataclass
@ProcessorStepRegistry.register(name="eo1_qwen_processor")
class EO1QwenProcessorStep(ComplementaryDataProcessorStep):
    # processor_name: str = "/mnt/inspurfs/evla2_t/eo-robotics/eo1_artifacts/Qwen2.5-VL-3B-Instruct"
    processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    _processor: Qwen2_5_VLProcessor | None = field(default=None, init=False, repr=False)
    _state_token_id: int | None = field(default=None, init=False, repr=False)
    _action_token_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._processor = Qwen2_5_VLProcessor.from_pretrained(self.processor_name)

        special_tokens = [
            ACTION_START_TOKEN,
            DEFAULT_ACTION_TOKEN,
            ACTION_END_TOKEN,
            STATE_START_TOKEN,
            DEFAULT_STATE_TOKEN,
            STATE_END_TOKEN,
            TASK_VLA_TOKEN,
        ]
        self._processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self._state_token_id = self._processor.tokenizer.convert_tokens_to_ids(DEFAULT_STATE_TOKEN)
        self._action_token_id = self._processor.tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN)

    def complementary_data(self, complementary_data):
        messages = complementary_data.pop("messages", None)
        if messages is None:
            raise ValueError("Messages are required for EO1QwenProcessorStep.")

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            do_resize=False,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        complementary_data["input_ids"] = inputs["input_ids"]
        complementary_data["pixel_values"] = inputs["pixel_values"]
        complementary_data["image_grid_thw"] = inputs["image_grid_thw"]

        complementary_data["attention_mask"] = inputs["attention_mask"]
        complementary_data["state_token_id"] = self._state_token_id
        complementary_data["action_token_id"] = self._action_token_id

        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step only converts the messages to the model input format.
        """
        return features


def make_eo1_pre_post_processors(
    config: EO1Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre/post processor pipelines for EO1."""

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        EO1ImageSmartResizeStep(
            input_features=config.input_features,
            image_min_pixels=config.image_min_pixels,
            image_max_pixels=config.image_max_pixels,
            image_resized_width=config.image_resized_width,
            image_resized_height=config.image_resized_height,
            size_factor=IMAGE_FACTOR,
        ),
        EO1ActionPaddingProcessorStep(max_action_dim=config.max_action_dim),
        EO1StatePaddingProcessorStep(max_state_dim=config.max_state_dim),
        EO1ConversationTemplateStep(input_features=config.input_features, chunk_size=config.chunk_size),
        EO1QwenProcessorStep(processor_name=config.vlm_base),
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
