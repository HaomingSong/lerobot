#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from lerobot.rewards.topreward.configuration_topreward import (
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX_TEMPLATE,
)
from lerobot.rewards.topreward.processor_topreward import (
    TOPREWARD_FEATURE_PREFIX,
    TOPRewardEncoderProcessorStep,
    _TRUE_ANSWER,
    _frames_to_pil,
    _video_to_numpy,
)
from lerobot.types import TransitionKey


def make_dummy_frames(batch_size: int, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Build deterministic uint8 BTHW video frames shaped like LeRobot image batches."""
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


def tensor_payload(value: torch.Tensor, *, full_values: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    cpu = value.detach().cpu()
    if full_values or cpu.numel() <= 200:
        payload["values"] = cpu.tolist()
    else:
        payload["first_32_values"] = cpu.flatten()[:32].tolist()
    return payload


def markdown_tensor(name: str, value: torch.Tensor, *, full_values: bool = False) -> str:
    payload = tensor_payload(value, full_values=full_values)
    lines = [
        f"- `{name}`: shape={payload['shape']}, dtype={payload['dtype']}, device={payload['device']}"
    ]
    if "values" in payload:
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload["values"], indent=2))
        lines.append("```")
    else:
        lines.append(f"  - first_32_values: `{payload['first_32_values']}`")
    return "\n".join(lines)


def capture_per_sample_inputs(
    step: TOPRewardEncoderProcessorStep,
    frames: torch.Tensor,
    tasks: list[str],
    *,
    add_chat_template: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(frames.shape[0]):
        frames_np = _video_to_numpy(frames[i], max_frames=step.max_frames)
        pil_frames = _frames_to_pil(frames_np)
        task = tasks[i]
        instruction_suffix = step.prompt_suffix_template.format(instruction=task)
        eos_token = step._processor.tokenizer.eos_token

        if add_chat_template:
            suffix_for_template = instruction_suffix.removesuffix(_TRUE_ANSWER).rstrip()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": step.fps},
                        {"type": "text", "text": f"{step.prompt_prefix}{suffix_for_template}"},
                    ],
                }
            ]
            prompt_chat = step._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = f"{prompt_chat}{_TRUE_ANSWER}"
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": step.fps},
                        {"type": "text", "text": step.prompt_prefix},
                    ],
                }
            ]
            prompt_chat = step._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if eos_token is not None:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = step._processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        records.append(
            {
                "sample_index": i,
                "task": task,
                "prompt_chat": prompt_chat,
                "full_text": full_text,
                "input_keys": list(inputs.keys()),
                "inputs": dict(inputs),
                "non_pad_length": int(inputs["attention_mask"].to(torch.long).sum(dim=1)[0].item()),
            }
        )
    return records


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

    per_sample = capture_per_sample_inputs(
        step, frames, tasks, add_chat_template=add_chat_template
    )
    transition = {
        TransitionKey.OBSERVATION: {"observation.images.top": frames},
        TransitionKey.COMPLEMENTARY_DATA: {"task": tasks},
    }
    encoded = step(transition)
    return {
        "frames_shape": list(frames.shape),
        "tasks": tasks,
        "per_sample": per_sample,
        "batch_observation": encoded[TransitionKey.OBSERVATION],
    }


def write_report(args: argparse.Namespace) -> Path:
    cases = {
        False: run_case(args, False),
        True: run_case(args, True),
    }

    lines: list[str] = [
        "# Qwen3-VL TOPReward Processor Intermediate Report",
        "",
        "## Runtime",
        "",
        f"- model_id: `{args.model_id}`",
        f"- batch_size: `{args.batch_size}`",
        f"- dummy frames: `(B,T,C,H,W) = {cases[False]['frames_shape']}`",
        f"- output_dir: `{args.output_dir}`",
        f"- HF_HOME: `{os.environ.get('HF_HOME', '')}`",
        f"- HF_HUB_CACHE: `{os.environ.get('HF_HUB_CACHE', '')}`",
        f"- proxy env present: HTTPS_PROXY={bool(os.environ.get('HTTPS_PROXY'))}, HTTP_PROXY={bool(os.environ.get('HTTP_PROXY'))}, https_proxy={bool(os.environ.get('https_proxy'))}, http_proxy={bool(os.environ.get('http_proxy'))}",
        "",
        "This report captures the exact per-sample `inputs` returned at the current TOPReward code path:",
        "",
        "```python",
        "inputs = self._processor(",
        "    text=[full_text],",
        "    images=image_inputs,",
        "    videos=video_inputs,",
        "    padding=True,",
        "    return_tensors=\"pt\",",
        ")",
        "```",
        "",
    ]

    for add_chat_template, case in cases.items():
        lines.extend(
            [
                f"## add_chat_template={add_chat_template}",
                "",
                f"- tasks: `{case['tasks']}`",
                "",
            ]
        )
        for record in case["per_sample"]:
            lines.extend(
                [
                    f"### Sample {record['sample_index']}",
                    "",
                    f"- task: `{record['task']}`",
                    f"- non_pad_length: `{record['non_pad_length']}`",
                    f"- exact `inputs.keys()`: `{record['input_keys']}`",
                    "",
                    "#### prompt_chat",
                    "",
                    "```text",
                    record["prompt_chat"],
                    "```",
                    "",
                    "#### full_text",
                    "",
                    "```text",
                    record["full_text"],
                    "```",
                    "",
                    "#### `inputs` from `self._processor(...)`",
                    "",
                ]
            )
            for key, value in record["inputs"].items():
                full_values = key in {"input_ids", "attention_mask", "mm_token_type_ids"}
                if isinstance(value, torch.Tensor):
                    lines.append(markdown_tensor(key, value, full_values=full_values))
                    lines.append("")
                else:
                    lines.append(f"- `{key}`: {type(value).__name__} `{value!r}`")
                    lines.append("")

        obs = case["batch_observation"]
        lines.extend(["### Final `TOPRewardEncoderProcessorStep.__call__()` Observation Keys", ""])
        top_keys = sorted(k for k in obs if k.startswith(TOPREWARD_FEATURE_PREFIX))
        lines.append(f"- keys: `{top_keys}`")
        lines.append("")
        for key in top_keys:
            value = obs[key]
            short = key.removeprefix(TOPREWARD_FEATURE_PREFIX)
            if isinstance(value, torch.Tensor):
                full_values = short in {"input_ids", "attention_mask", "mm_token_type_ids", "prompt_length"}
                lines.append(markdown_tensor(key, value, full_values=full_values))
                lines.append("")
            else:
                lines.append(f"- `{key}`: {type(value).__name__} `{value!r}`")
                lines.append("")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "qwen3vl_topreward_processor_intermediate_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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
    args = parser.parse_args()

    path = write_report(args)
    print(f"Wrote report to: {path}", flush=True)


if __name__ == "__main__":
    main()
