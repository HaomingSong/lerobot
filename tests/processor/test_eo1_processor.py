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
"""Integration-style tests for the EO1 processor pipeline."""

from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy_config
from lerobot.policies.eo1.processor_eo1 import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_STATE_TOKEN,
    EO1ActionPaddingProcessorStep,
    EO1ConversationTemplateStep,
    EO1ImageSmartResizeStep,
    EO1QwenProcessorStep,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    SYSTEM_MESSAGE,
    TASK_VLA_TOKEN,
    make_eo1_pre_post_processors,
)
from lerobot.processor import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "eo1_artifacts" / "libero_hf"
DEFAULT_QWEN_ROOT = WORKSPACE_ROOT / "eo1_artifacts" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_DATASET_REPO_ID = "HuggingFaceVLA/libero"


def _get_env_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser() if value else default


def _get_env_int(env_name: str, default: int) -> int:
    value = os.getenv(env_name)
    return int(value) if value else default


def _get_env_optional_int(env_name: str) -> int | None:
    value = os.getenv(env_name)
    return int(value) if value else None


DEBUG_TENSOR_CONTENT_KEYS = frozenset({"action_is_pad", "index", "task_index", "episode_index"})


def _format_debug_value(value: Any, indent: int = 0, path: tuple[str, ...] = ()) -> str:
    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        tensor_summary = (
            f"{prefix}Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, numel={value.numel()})"
        )
        if path and path[-1] in DEBUG_TENSOR_CONTENT_KEYS:
            return f"{tensor_summary}\n{prefix}{value}"
        return tensor_summary
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}"
        lines = [f"{prefix}{{"]
        for key, sub_value in value.items():
            lines.append(f"{prefix}  {key}:")
            lines.append(_format_debug_value(sub_value, indent + 4, (*path, str(key))))
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = [f"{prefix}["]
        for index, item in enumerate(value):
            lines.append(f"{prefix}  [{index}]")
            lines.append(_format_debug_value(item, indent + 4, (*path, f"[{index}]")))
        lines.append(f"{prefix}]")
        return "\n".join(lines)
    if isinstance(value, tuple):
        if not value:
            return f"{prefix}()"
        lines = [f"{prefix}("]
        for index, item in enumerate(value):
            lines.append(f"{prefix}  ({index})")
            lines.append(_format_debug_value(item, indent + 4, (*path, f"({index})")))
        lines.append(f"{prefix})")
        return "\n".join(lines)
    return f"{prefix}{repr(value)}"


def _print_debug_snapshot(title: str, value: Any) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")
    print(_format_debug_value(value))
    print(f"{'=' * 24} END {title} {'=' * 20}\n")


def _register_debug_hooks(preprocessor) -> tuple[Any, Any]:
    step_names = [step.__class__.__name__ for step in preprocessor.steps]

    def before_hook(step_index, transition):
        step_name = step_names[step_index]
        _print_debug_snapshot(f"BEFORE STEP {step_index}: {step_name}", transition)

    def after_hook(step_index, transition):
        step_name = step_names[step_index]
        _print_debug_snapshot(f"AFTER STEP {step_index}: {step_name}", transition)

    preprocessor.register_before_step_hook(before_hook)
    preprocessor.register_after_step_hook(after_hook)
    return before_hook, after_hook


def _decode_processed_input_ids(preprocessor, processed_batch: dict[str, Any]) -> list[str]:
    qwen_step = next(step for step in preprocessor.steps if isinstance(step, EO1QwenProcessorStep))
    if qwen_step._processor is None:
        raise ValueError("EO1QwenProcessorStep processor is not initialized.")

    input_ids = processed_batch["input_ids"].detach().cpu()
    attention_mask = processed_batch["attention_mask"].detach().cpu().bool()
    tokenizer = qwen_step._processor.tokenizer

    decoded_texts = []
    for token_ids, mask in zip(input_ids, attention_mask, strict=True):
        visible_token_ids = token_ids[mask].tolist()
        decoded_texts.append(
            tokenizer.decode(
                visible_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    return decoded_texts


def _assert_decoded_input_texts(decoded_texts: list[str], raw_batch: dict[str, Any], chunk_size: int) -> None:
    expected_action_text = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN * chunk_size}{ACTION_END_TOKEN}"
    expected_state_text = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"

    for decoded_text, task in zip(decoded_texts, raw_batch["task"], strict=True):
        assert SYSTEM_MESSAGE in decoded_text
        assert task in decoded_text
        assert TASK_VLA_TOKEN in decoded_text
        assert expected_state_text in decoded_text
        assert expected_action_text in decoded_text


def _build_eo1_train_config(
    dataset_root: Path,
    qwen_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=os.getenv("EO1_PROCESSOR_DATASET_REPO_ID", DEFAULT_DATASET_REPO_ID),
            root=str(dataset_root),
        ),
        policy=make_policy_config(
            "eo1",
            push_to_hub=False,
            device=device,
            vlm_base=str(qwen_root),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    cfg.validate()
    return cfg


def _build_dataloader(cfg: TrainPipelineConfig, dataset):
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=cfg.policy.device.startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def _assert_first_batch_messages(preprocessor, raw_batch: dict, camera_keys: list[str], chunk_size: int) -> None:
    conversation_step_index = next(
        index
        for index, step in enumerate(preprocessor.steps, start=1)
        if isinstance(step, EO1ConversationTemplateStep)
    )
    conversation_transition = list(preprocessor.step_through(deepcopy(raw_batch)))[conversation_step_index]
    messages = conversation_transition[TransitionKey.COMPLEMENTARY_DATA.value]["messages"]

    assert len(messages) == len(raw_batch["task"])
    expected_action_text = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN * chunk_size}{ACTION_END_TOKEN}"

    for sample_messages, task in zip(messages, raw_batch["task"], strict=True):
        assert len(sample_messages) == 3
        assert sample_messages[0]["role"] == "system"
        assert sample_messages[0]["content"][0]["text"] == SYSTEM_MESSAGE

        assert sample_messages[1]["role"] == "user"
        user_content = sample_messages[1]["content"]
        image_entries = [entry for entry in user_content if entry["type"] == "image"]
        text_entries = [entry["text"] for entry in user_content if entry["type"] == "text"]

        assert len(image_entries) == len(camera_keys)
        assert len(text_entries) == 1
        assert text_entries[0] == (
            f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}{task}{TASK_VLA_TOKEN}"
        )

        assert sample_messages[2]["role"] == "assistant"
        assert sample_messages[2]["content"][0]["text"] == expected_action_text


def _assert_batch_not_mutated(raw_batch: dict, original_batch: dict) -> None:
    for key, original_value in original_batch.items():
        current_value = raw_batch[key]
        if isinstance(original_value, torch.Tensor):
            assert torch.equal(current_value, original_value), f"Batch tensor '{key}' was mutated by preprocessing."
        else:
            assert current_value == original_value, f"Batch field '{key}' was mutated by preprocessing."


def _assert_processed_batch(
    processed_batch: dict,
    raw_batch: dict,
    dataset_stats: dict,
    cfg: TrainPipelineConfig,
    camera_keys: list[str],
    resized_features: dict[str, torch.Size | tuple[int, ...]],
) -> None:
    batch_size = len(raw_batch["task"])
    raw_action = raw_batch[ACTION]
    raw_state = raw_batch[OBS_STATE]

    assert processed_batch[ACTION].shape == (*raw_action.shape[:-1], cfg.policy.max_action_dim)
    assert processed_batch[OBS_STATE].shape == (*raw_state.shape[:-1], cfg.policy.max_state_dim)
    assert torch.equal(processed_batch["action_is_pad"], raw_batch["action_is_pad"])
    assert processed_batch["task"] == raw_batch["task"]

    action_mean = torch.as_tensor(
        dataset_stats[ACTION]["mean"], dtype=raw_action.dtype, device=raw_action.device
    )
    action_std = torch.as_tensor(
        dataset_stats[ACTION]["std"], dtype=raw_action.dtype, device=raw_action.device
    )
    expected_action = (raw_action - action_mean) / (action_std + 1e-8)

    state_mean = torch.as_tensor(
        dataset_stats[OBS_STATE]["mean"], dtype=raw_state.dtype, device=raw_state.device
    )
    state_std = torch.as_tensor(
        dataset_stats[OBS_STATE]["std"], dtype=raw_state.dtype, device=raw_state.device
    )
    expected_state = (raw_state - state_mean) / (state_std + 1e-8)

    torch.testing.assert_close(
        processed_batch[ACTION][..., : raw_action.shape[-1]].cpu(),
        expected_action.cpu(),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        processed_batch[OBS_STATE][..., : raw_state.shape[-1]].cpu(),
        expected_state.cpu(),
        rtol=1e-4,
        atol=1e-4,
    )

    assert torch.count_nonzero(processed_batch[ACTION][..., raw_action.shape[-1] :]).item() == 0
    assert torch.count_nonzero(processed_batch[OBS_STATE][..., raw_state.shape[-1] :]).item() == 0

    for key in camera_keys:
        expected_shape = tuple(resized_features[key])
        assert processed_batch[key].shape == (batch_size, *expected_shape)
        assert torch.isfinite(processed_batch[key]).all()

    input_ids = processed_batch["input_ids"]
    attention_mask = processed_batch["attention_mask"]
    image_grid_thw = processed_batch["image_grid_thw"]
    pixel_values = processed_batch["pixel_values"]

    assert input_ids.shape[0] == batch_size
    assert attention_mask.shape == input_ids.shape
    assert image_grid_thw.shape == (batch_size * len(camera_keys), 3)
    assert pixel_values.ndim == 2
    assert pixel_values.shape[0] > 0
    assert torch.isfinite(pixel_values).all()

    state_token_id = processed_batch["state_token_id"]
    action_token_id = processed_batch["action_token_id"]
    assert torch.all((input_ids == state_token_id).sum(dim=1) == 1)
    assert torch.all((input_ids == action_token_id).sum(dim=1) == cfg.policy.chunk_size)


def test_eo1_action_padding_skips_missing_action():
    step = EO1ActionPaddingProcessorStep(max_action_dim=32)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 8)},
        TransitionKey.ACTION: None,
    }

    processed_transition = step(transition)

    assert processed_transition[TransitionKey.ACTION] is None
    torch.testing.assert_close(
        processed_transition[TransitionKey.OBSERVATION][OBS_STATE],
        transition[TransitionKey.OBSERVATION][OBS_STATE],
    )


def test_eo1_processor_pipeline_with_libero_dataset():
    dataset_root = _get_env_path("EO1_PROCESSOR_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    qwen_root = _get_env_path("EO1_PROCESSOR_QWEN_ROOT", DEFAULT_QWEN_ROOT)

    if not dataset_root.exists():
        pytest.skip(f"EO1 processor dataset not found: {dataset_root}")
    if not qwen_root.exists():
        pytest.skip(f"EO1 Qwen checkpoint not found: {qwen_root}")

    batch_size = _get_env_int("EO1_PROCESSOR_BATCH_SIZE", 64)
    num_workers = _get_env_int("EO1_PROCESSOR_NUM_WORKERS", 8)
    max_batches = _get_env_optional_int("EO1_PROCESSOR_MAX_BATCHES")
    progress_every = _get_env_int("EO1_PROCESSOR_PROGRESS_EVERY", 200)
    device = os.getenv("EO1_PROCESSOR_DEVICE", "cpu")

    cfg = _build_eo1_train_config(
        dataset_root,
        qwen_root,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    dataset = make_dataset(cfg)
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.policy.output_features = {key: value for key, value in features.items() if key == ACTION}
    cfg.policy.input_features = {key: value for key, value in features.items() if key != ACTION}
    cfg.policy.validate_features()

    preprocessor, _ = make_eo1_pre_post_processors(cfg.policy, dataset.meta.stats)
    resize_step = next(step for step in preprocessor.steps if isinstance(step, EO1ImageSmartResizeStep))
    resized_features = {key: feature.shape for key, feature in resize_step._resized_features.items()}
    dataloader = _build_dataloader(cfg, dataset)

    total_frames = 0
    total_batches = 0
    expected_total_batches = min(
        math.ceil(len(dataset) / batch_size),
        max_batches if max_batches is not None else math.ceil(len(dataset) / batch_size),
    )
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(dataloader, start=1):
            if batch_index == 1:
                batch_before_processing = deepcopy(raw_batch)
                _assert_first_batch_messages(
                    preprocessor,
                    raw_batch,
                    dataset.meta.camera_keys,
                    cfg.policy.chunk_size,
                )
                processed_batch = preprocessor(raw_batch)
                decoded_input_texts = _decode_processed_input_ids(preprocessor, processed_batch)
                _assert_decoded_input_texts(decoded_input_texts, raw_batch, cfg.policy.chunk_size)
                _assert_batch_not_mutated(raw_batch, batch_before_processing)
            else:
                processed_batch = preprocessor(raw_batch)

            _assert_processed_batch(
                processed_batch,
                raw_batch,
                dataset.meta.stats,
                cfg,
                dataset.meta.camera_keys,
                resized_features,
            )

            total_frames += len(raw_batch["task"])
            total_batches = batch_index

            if progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0):
                elapsed = time.perf_counter() - start_time
                print(
                    "[eo1-processor-test] "
                    f"batches={batch_index}/{expected_total_batches} "
                    f"frames={total_frames} elapsed_s={elapsed:.2f}"
                )

            if max_batches is not None and batch_index >= max_batches:
                break

    elapsed = time.perf_counter() - start_time
    print(
        "[eo1-processor-test] summary "
        f"dataset_frames={len(dataset)} processed_frames={total_frames} "
        f"processed_batches={total_batches} elapsed_s={elapsed:.2f}"
    )

    assert total_batches > 0
    if max_batches is None:
        assert total_frames == len(dataset)


def test_eo1_processor_pipeline_debug_visualization():
    dataset_root = _get_env_path("EO1_PROCESSOR_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    qwen_root = _get_env_path("EO1_PROCESSOR_QWEN_ROOT", DEFAULT_QWEN_ROOT)

    if not dataset_root.exists():
        pytest.skip(f"EO1 processor dataset not found: {dataset_root}")
    if not qwen_root.exists():
        pytest.skip(f"EO1 Qwen checkpoint not found: {qwen_root}")

    torch.set_printoptions(profile="full", linewidth=200)

    cfg = _build_eo1_train_config(
        dataset_root,
        qwen_root,
        batch_size=1,
        num_workers=0,
        device=os.getenv("EO1_PROCESSOR_DEVICE", "cpu"),
    )
    dataset = make_dataset(cfg)
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.policy.output_features = {key: value for key, value in features.items() if key == ACTION}
    cfg.policy.input_features = {key: value for key, value in features.items() if key != ACTION}
    cfg.policy.validate_features()

    preprocessor, _ = make_eo1_pre_post_processors(cfg.policy, dataset.meta.stats)
    resize_step = next(step for step in preprocessor.steps if isinstance(step, EO1ImageSmartResizeStep))
    resized_features = {key: feature.shape for key, feature in resize_step._resized_features.items()}
    raw_batch = next(iter(_build_dataloader(cfg, dataset)))
    batch_before_processing = deepcopy(raw_batch)

    _print_debug_snapshot("RAW BATCH", raw_batch)
    initial_transition = preprocessor.to_transition(deepcopy(raw_batch))
    _print_debug_snapshot("INITIAL ENVTRANSITION", initial_transition)

    before_hook, after_hook = _register_debug_hooks(preprocessor)
    try:
        processed_batch = preprocessor(raw_batch)
    finally:
        preprocessor.unregister_before_step_hook(before_hook)
        preprocessor.unregister_after_step_hook(after_hook)

    decoded_input_texts = _decode_processed_input_ids(preprocessor, processed_batch)
    _assert_decoded_input_texts(decoded_input_texts, raw_batch, cfg.policy.chunk_size)

    processed_batch_for_debug = dict(processed_batch)
    processed_batch_for_debug["decoded_input_text"] = decoded_input_texts
    _print_debug_snapshot("PROCESSED BATCH", processed_batch_for_debug)

    _assert_batch_not_mutated(raw_batch, batch_before_processing)
    _assert_processed_batch(
        processed_batch,
        raw_batch,
        dataset.meta.stats,
        cfg,
        dataset.meta.camera_keys,
        resized_features,
    )
