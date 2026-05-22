#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from lerobot.rewards.topreward.configuration_topreward import (
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX_TEMPLATE,
)
from lerobot.rewards.topreward.processor_topreward import (
    TOPREWARD_FEATURE_PREFIX,
    TOPRewardEncoderProcessorStep,
)
from lerobot.types import TransitionKey


def make_dummy_frames(batch_size: int, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Build deterministic uint8 BCHW video frames shaped like LeRobot image batches."""
    y = torch.arange(height, dtype=torch.uint8).view(1, 1, height, 1)
    x = torch.arange(width, dtype=torch.uint8).view(1, 1, 1, width)
    frames = []
    for b in range(batch_size):
        per_sample = []
        for t in range(num_frames):
            r = (x + 17 * t + 31 * b).expand(1, 1, height, width)
            g = (y + 11 * t + 13 * b).expand(1, 1, height, width)
            bl = ((x // 2 + y // 3) + 23 * t + 7 * b).expand(1, 1, height, width)
            per_sample.append(torch.cat([r, g, bl], dim=1).squeeze(0))
        frames.append(torch.stack(per_sample, dim=0))
    return torch.stack(frames, dim=0)


def tensor_summary(value: torch.Tensor, seq_len: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": "Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sequence_aligned": bool(value.ndim >= 2 and value.shape[0] > 0 and value.shape[1] == seq_len),
    }
    if value.numel() <= 20:
        summary["values"] = value.detach().cpu().tolist()
    elif value.ndim >= 2 and value.shape[0] > 0:
        summary["first_row_prefix"] = value[0].flatten()[:20].detach().cpu().tolist()
    return summary


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    input_ids = batch[f"{TOPREWARD_FEATURE_PREFIX}input_ids"]
    seq_len = int(input_ids.shape[1])
    out: dict[str, Any] = {}
    for key, value in sorted(batch.items()):
        if key.startswith(TOPREWARD_FEATURE_PREFIX):
            short_key = key.removeprefix(TOPREWARD_FEATURE_PREFIX)
            if isinstance(value, torch.Tensor):
                out[short_key] = tensor_summary(value, seq_len)
            else:
                out[short_key] = {"type": type(value).__name__, "repr": repr(value)}
    return out


def run_case(args: argparse.Namespace, add_chat_template: bool) -> dict[str, Any]:
    frames = make_dummy_frames(args.batch_size, args.num_frames, args.height, args.width)
    tasks = [f"pick up the red cube sample {idx}" for idx in range(args.batch_size)]
    step = TOPRewardEncoderProcessorStep(
        vlm_name=args.model_id,
        image_key="observation.images.top",
        task_key="task",
        default_task=None,
        max_frames=None,
        fps=args.fps,
        prompt_prefix=DEFAULT_PROMPT_PREFIX,
        prompt_suffix_template=DEFAULT_PROMPT_SUFFIX_TEMPLATE,
        add_chat_template=add_chat_template,
        max_length=args.max_length,
    )
    transition = {
        TransitionKey.OBSERVATION: {"observation.images.top": frames},
        TransitionKey.COMPLEMENTARY_DATA: {"task": tasks},
    }
    encoded = step(transition)
    obs = encoded[TransitionKey.OBSERVATION]
    summary = summarize_batch(obs)
    attention_mask = obs[f"{TOPREWARD_FEATURE_PREFIX}attention_mask"]
    summary["_case"] = {
        "add_chat_template": add_chat_template,
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "frame_shape": [args.height, args.width],
        "non_pad_lengths": attention_mask.to(torch.long).sum(dim=1).tolist(),
        "prompt_lengths": obs[f"{TOPREWARD_FEATURE_PREFIX}prompt_length"].tolist(),
        "tasks": tasks,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--download-full-weights", action="store_true")
    parser.add_argument("--download-max-workers", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_snapshot = {
        key: ("SET" if os.environ.get(key) else "unset")
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "HF_HOME", "HF_HUB_CACHE")
    }
    print("Environment:", json.dumps(env_snapshot, indent=2), flush=True)

    if args.download_full_weights:
        print(f"Downloading full snapshot for {args.model_id} ...", flush=True)
        snapshot_path = snapshot_download(
            repo_id=args.model_id,
            local_dir=None,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=args.download_max_workers,
        )
        print(f"Snapshot downloaded to: {snapshot_path}", flush=True)

    combined = {
        "model_id": args.model_id,
        "env": env_snapshot,
        "cases": {
            "add_chat_template_false": run_case(args, False),
            "add_chat_template_true": run_case(args, True),
        },
    }
    out_path = output_dir / "qwen3vl_topreward_processor_contract.json"
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2), flush=True)
    print(f"Wrote JSON summary to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
