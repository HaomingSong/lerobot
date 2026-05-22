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

"""TOPReward pre/post processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    policy_action_to_transition,
)
from lerobot.rewards.topreward.configuration_topreward import (
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX_TEMPLATE,
    TOPRewardConfig,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

TOPREWARD_FEATURE_PREFIX = f"{OBS_PREFIX}topreward."

_TRUE_ANSWER = "True"
_QWEN_ASSISTANT_GENERATION_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

TOPREWARD_VLM_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "labels",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
)
TOPREWARD_INPUT_KEYS = TOPREWARD_VLM_INPUT_KEYS


def _video_to_numpy(video: Tensor, *, max_frames: int | None) -> np.ndarray:
    """Convert one trajectory tensor to a ``(T, H, W, C) uint8`` numpy array."""
    if max_frames is not None:
        video = video[-max_frames:]
    if video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Expected channel dim of size 1 or 3, got shape {tuple(video.shape)}")

    array = video.detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _frames_to_pil(frames: np.ndarray) -> list[Image.Image]:
    """Convert ``(T, H, W, C)`` uint8 frames to a list of PIL images."""
    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C) frames; got shape {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return [Image.fromarray(frames[i]) for i in range(frames.shape[0])]


def _expand_tasks(task: Any, *, batch_size: int, default: str | None) -> list[str]:
    if task is None:
        task = default
    if task is None:
        raise KeyError("TOPReward expected a task description in complementary data")
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, tuple):
        task = list(task)
    if not (isinstance(task, list) and all(isinstance(item, str) for item in task)):
        raise TypeError(f"TOPReward task must be a string or list of strings, got {type(task)}")
    if len(task) == 1 and batch_size > 1:
        return task * batch_size
    if len(task) != batch_size:
        raise ValueError(f"Expected {batch_size} tasks, got {len(task)}")
    return task


@dataclass
@ProcessorStepRegistry.register(name="topreward_encoder")
class TOPRewardEncoderProcessorStep(ProcessorStep):
    """Encode raw frames + task into Qwen-VL tensors for the TOPReward model.

    Loads a :class:`~transformers.AutoProcessor` matching ``vlm_name`` and
    builds the full chat prompt including the instruction suffix. The
    resulting ``input_ids``, ``attention_mask``, label tensor, and vision
    tensors are written under the ``observation.topreward.*`` namespace so
    the model can forward without re-tokenising or rebuilding labels.

    At call time the step reads:

    - ``observation[image_key]``: ``(B, T, C, H, W)`` or ``(B, C, H, W)`` frames.
    - ``complementary_data[task_key]``: a string or list of strings.

    and writes ``observation[f"{TOPREWARD_FEATURE_PREFIX}<name>"]`` for the
    Qwen-VL tensors plus ``labels``.
    """

    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 16
    fps: float = 2.0
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX
    prompt_suffix_template: str = DEFAULT_PROMPT_SUFFIX_TEMPLATE
    add_chat_template: bool = False
    max_length: int = 32768

    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_package("transformers", extra="topreward")
        require_package("qwen-vl-utils", extra="topreward", import_name="qwen_vl_utils")
        self._processor = AutoProcessor.from_pretrained(self.vlm_name, trust_remote_code=True)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if not isinstance(observation, dict):
            raise ValueError("TOPRewardEncoderProcessorStep requires an observation dict")

        if self.image_key not in observation:
            raise KeyError(f"TOPReward expected image key {self.image_key!r} in observation")

        frames = observation[self.image_key]
        tensor = frames.detach().cpu() if isinstance(frames, Tensor) else torch.as_tensor(frames)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 5:
            raise ValueError(
                f"Expected TOPReward frames with shape (B,C,H,W) or (B,T,C,H,W); got {tuple(tensor.shape)}"
            )

        batch_size = tensor.shape[0]
        tasks = _expand_tasks(
            complementary.get(self.task_key, self.default_task),
            batch_size=batch_size,
            default=self.default_task,
        )

        encoded = self._encode_batch(tensor, tasks)

        new_observation = dict(observation)
        for key, value in encoded.items():
            new_observation[f"{TOPREWARD_FEATURE_PREFIX}{key}"] = value

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _encode_batch(self, tensor: Tensor, tasks: list[str]) -> dict[str, Any]:
        """Tokenise a batch of (frames, task) pairs into Qwen-VL tensors.

        The chat messages are built per sample, but the Qwen processor is
        called once for the full batch. Text tensors are left padded so the
        final answer token is always at ``[:, -1]``; labels therefore mask
        everything except that final token.
        """
        templated_messages = []

        for i, task in enumerate(tasks):
            frames_np = _video_to_numpy(tensor[i], max_frames=self.max_frames)
            pil_frames = _frames_to_pil(frames_np)

            instruction_suffix = self.prompt_suffix_template.format(instruction=task)
            templated_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": pil_frames, "fps": self.fps},
                            {"type": "text", "text": f"{self.prompt_prefix}{instruction_suffix}"},
                        ],
                    }
                ]
            )

        # TODO: 这里把tokenizer拿出来设置padding side的方式太愚蠢了，apply_chat_template应该可以直接设置padding side=left，请你仔细查看
        tokenizer = self._processor.tokenizer
        original_padding_side = getattr(tokenizer, "padding_side", None)
        if original_padding_side is not None:
            tokenizer.padding_side = "left"
        try:
            encoded = self._processor.apply_chat_template(
                templated_messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
                return_mm_token_type_ids=True,  # TODO: 这里有必要显式添加吗？
            )
        finally:
            if original_padding_side is not None:
                tokenizer.padding_side = original_padding_side

        # NOTE: 这里为什么要手动转dict？不能直接用key或者.索引吗
        result = dict(encoded)
        if not self.add_chat_template:
            trailer_token_ids = self._encode_one_or_more_tokens(_QWEN_ASSISTANT_GENERATION_SUFFIX)
            result = self._remove_sequence_suffix(result, trailer_token_ids)

        answer_text = _TRUE_ANSWER if self.add_chat_template else f" {_TRUE_ANSWER}"
        answer_token_id = self._encode_single_token(answer_text)
        result = self._append_answer_token_and_labels(result, answer_token_id)

        input_len = int(result["input_ids"].shape[-1])
        if input_len > self.max_length:
            raise ValueError(
                f"TOPReward input length {input_len} exceeds max_length "
                f"{self.max_length}; lower `max_frames` or raise `max_length`."
            )
        return result

    # TODO: 移除_encode_one_or_more_tokens与_encode_single_token，不需要异常诊断，调用tokenizer编码直接放在_encode_batch里面,
    def _encode_one_or_more_tokens(self, text: str) -> torch.Tensor:
        token_ids = self._processor.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"TOPReward expected {text!r} to tokenize to at least one token")
        # TODO: 请你确认能否直接让tokenizer return torch.long tensor，可以省略这里的转换步骤
        return torch.tensor(token_ids, dtype=torch.long)

    def _encode_single_token(self, text: str) -> int:
        token_ids = self._processor.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"TOPReward expected {text!r} to tokenize to one token, got token ids {token_ids}"
            )
        return int(token_ids[0])

    @staticmethod
    def _remove_sequence_suffix(encoded: dict[str, Any], suffix_token_ids: Tensor) -> dict[str, Any]:
        input_ids = encoded["input_ids"]
        suffix = suffix_token_ids.to(device=input_ids.device, dtype=input_ids.dtype)
        suffix_len = int(suffix.numel())
        if suffix_len >= input_ids.shape[-1]:
            raise ValueError("TOPReward cannot remove the assistant generation prompt from an empty input")

        actual_suffix = input_ids[:, -suffix_len:]
        expected_suffix = suffix.unsqueeze(0).expand_as(actual_suffix)
        if not torch.equal(actual_suffix, expected_suffix):
            raise ValueError(
                "TOPReward expected the Qwen chat template to end with "
                f"{_QWEN_ASSISTANT_GENERATION_SUFFIX!r}, but the tokenized suffix did not match."
            )

        result: dict[str, Any] = {}
        sequence_shape = input_ids.shape
        for key, value in encoded.items():
            if isinstance(value, Tensor) and value.shape == sequence_shape:
                result[key] = value[:, :-suffix_len]
            else:
                result[key] = value
        return result

    @staticmethod
    def _append_answer_token_and_labels(encoded: dict[str, Any], answer_token_id: int) -> dict[str, Any]:
        input_ids = encoded["input_ids"]
        batch_size = input_ids.shape[0]
        answer_ids = torch.full(
            (batch_size, 1),
            answer_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        result: dict[str, Any] = {}
        sequence_shape = input_ids.shape
        for key, value in encoded.items():
            if isinstance(value, Tensor) and value.shape == sequence_shape:
                if key == "input_ids":
                    suffix = answer_ids
                elif key == "attention_mask":
                    suffix = torch.ones((batch_size, 1), dtype=value.dtype, device=value.device)
                else:
                    suffix = torch.zeros((batch_size, 1), dtype=value.dtype, device=value.device)
                result[key] = torch.cat([value, suffix], dim=1)
            else:
                result[key] = value

        labels = torch.full_like(result["input_ids"], -100)
        labels[:, -1] = result["input_ids"][:, -1]
        result["labels"] = labels
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "vlm_name": self.vlm_name,
            "image_key": self.image_key,
            "task_key": self.task_key,
            "default_task": self.default_task,
            "max_frames": self.max_frames,
            "fps": self.fps,
            "prompt_prefix": self.prompt_prefix,
            "prompt_suffix_template": self.prompt_suffix_template,
            "add_chat_template": self.add_chat_template,
            "max_length": self.max_length,
        }


def make_topreward_pre_post_processors(
    config: TOPRewardConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pipeline that pre-encodes frames + task into Qwen-VL tensors.

    The preprocessor adds a batch dimension if needed, runs TOPReward's
    encoder (which tokenises the full prompt and emits ``labels``),
    and moves everything to the configured device. The postprocessor is
    the identity since TOPReward outputs a single reward tensor.
    """
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            TOPRewardEncoderProcessorStep(
                vlm_name=config.vlm_name,
                image_key=config.image_key,
                task_key=config.task_key,
                default_task=config.default_task,
                max_frames=config.max_frames,
                fps=config.fps,
                prompt_prefix=config.prompt_prefix,
                prompt_suffix_template=config.prompt_suffix_template,
                add_chat_template=config.add_chat_template,
                max_length=config.max_input_length,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor
