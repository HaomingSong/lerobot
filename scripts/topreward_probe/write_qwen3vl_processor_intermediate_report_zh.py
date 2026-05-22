#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from write_qwen3vl_processor_intermediate_report import run_case
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX


def inline_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ": "))


def preview_flat_tensor(value: torch.Tensor, *, n: int = 10) -> dict[str, Any]:
    cpu = value.detach().cpu()
    flat = cpu.flatten()
    if flat.numel() <= 2 * n:
        return {"values": flat.tolist()}
    return {"first10": flat[:n].tolist(), "last10": flat[-n:].tolist()}


def preview_rows(value: torch.Tensor, *, n: int = 10) -> list[dict[str, Any]]:
    cpu = value.detach().cpu()
    if cpu.ndim == 0:
        return [{"row": 0, "values": [cpu.item()]}]
    if cpu.ndim == 1:
        return [{"row": 0, **preview_flat_tensor(cpu, n=n)}]
    return [{"row": i, **preview_flat_tensor(cpu[i], n=n)} for i in range(cpu.shape[0])]


def tensor_block(name: str, value: torch.Tensor, *, row_preview: bool) -> list[str]:
    lines = [
        f"- `{name}`: shape={list(value.shape)}, dtype={value.dtype}, device={value.device}"
    ]
    if row_preview:
        lines.append(f"  - batch 行预览: `{inline_json(preview_rows(value))}`")
    else:
        lines.append(f"  - 展平预览: `{inline_json(preview_flat_tensor(value))}`")
    return lines


def raw_input_block(records: list[dict[str, Any]], key: str) -> list[str]:
    values = [record["inputs"][key] for record in records]
    lines = [
        f"- `{key}`:",
        f"  - 每个 batch item 的 shape: `{inline_json([list(value.shape) for value in values])}`",
        f"  - dtype: `{values[0].dtype}`",
    ]
    if key in {"input_ids", "attention_mask", "mm_token_type_ids", "video_grid_thw"}:
        previews = [
            {"batch_index": i, **preview_flat_tensor(value)}
            for i, value in enumerate(values)
        ]
        lines.append(f"  - batch 预览: `{inline_json(previews)}`")
    else:
        previews = [
            {"batch_index": i, **preview_flat_tensor(value)}
            for i, value in enumerate(values)
        ]
        lines.append(f"  - 展平预览: `{inline_json(previews)}`")
    return lines


def write_report(args: argparse.Namespace) -> Path:
    cases = {
        False: run_case(args, False),
        True: run_case(args, True),
    }

    lines: list[str] = [
        "# Qwen3-VL TOPReward Processor 中间变量报告",
        "",
        "## 实验设置",
        "",
        f"- 模型: `{args.model_id}`",
        f"- batch size: `{args.batch_size}`",
        f"- dummy video frames: `(B,T,C,H,W) = {cases[False]['frames_shape']}`",
        f"- 输出目录: `{args.output_dir}`",
        f"- HF_HOME: `{os.environ.get('HF_HOME', '')}`",
        f"- HF_HUB_CACHE: `{os.environ.get('HF_HUB_CACHE', '')}`",
        f"- proxy env present: HTTPS_PROXY={bool(os.environ.get('HTTPS_PROXY'))}, HTTP_PROXY={bool(os.environ.get('HTTP_PROXY'))}, https_proxy={bool(os.environ.get('https_proxy'))}, http_proxy={bool(os.environ.get('http_proxy'))}",
        "",
        "本报告记录 `processor_topreward.py` 中当前路径下这一行返回的原始 `inputs`：",
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
        "说明：当前实现仍是在 `_encode_batch()` 内逐条样本调用 `_processor(text=[full_text], ...)`，所以这里把每次调用的结果按 batch 维度合并展示；不再按 Sample 0 / Sample 1 分节。长序列 tensor 仅展示每个 batch row 的前 10 个和后 10 个值。",
        "",
    ]

    for add_chat_template, case in cases.items():
        records = case["per_sample"]
        input_keys = records[0]["input_keys"]
        lines.extend(
            [
                f"## add_chat_template={add_chat_template}",
                "",
                f"- tasks: `{inline_json(case['tasks'])}`",
                f"- `_processor(...)` 原始 `inputs.keys()`: `{inline_json(input_keys)}`",
                f"- 非 padding 长度: `{inline_json([record['non_pad_length'] for record in records])}`",
                "",
                "### 输入 Processor 前的 prompt_chat",
                "",
                "```json",
                inline_json(
                    [
                        {
                            "batch_index": i,
                            "task": record["task"],
                            "prompt_chat": record["prompt_chat"],
                        }
                        for i, record in enumerate(records)
                    ]
                ),
                "```",
                "",
                "### 输入 Processor 前的 full_text",
                "",
                "```json",
                inline_json(
                    [
                        {
                            "batch_index": i,
                            "task": record["task"],
                            "full_text": record["full_text"],
                        }
                        for i, record in enumerate(records)
                    ]
                ),
                "```",
                "",
                "### `_processor(...)` 原始 inputs",
                "",
            ]
        )

        for key in input_keys:
            lines.extend(raw_input_block(records, key))
            lines.append("")

        obs = case["batch_observation"]
        top_keys = sorted(k for k in obs if k.startswith(TOPREWARD_FEATURE_PREFIX))
        lines.extend(
            [
                "### `TOPRewardEncoderProcessorStep.__call__()` 聚合后的 batch",
                "",
                f"- observation keys: `{inline_json(top_keys)}`",
                "",
            ]
        )
        for key in top_keys:
            value = obs[key]
            short_key = key.removeprefix(TOPREWARD_FEATURE_PREFIX)
            if isinstance(value, torch.Tensor):
                row_preview = short_key in {
                    "input_ids",
                    "attention_mask",
                    "mm_token_type_ids",
                    "prompt_length",
                    "video_grid_thw",
                }
                lines.extend(tensor_block(key, value, row_preview=row_preview))
                lines.append("")
            else:
                lines.append(f"- `{key}`: `{value!r}`")
                lines.append("")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "qwen3vl_topreward_processor_intermediate_report_zh.md"
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
