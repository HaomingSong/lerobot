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
"""Convert a DCP-format checkpoint into a distributable safetensors model, offline.

Runs single-process (no GPUs, no process group). The checkpoint comes either from a local
directory or from a Hub training repo:

```bash
lerobot-convert-dcp --checkpoint_dir=outputs/train/run/checkpoints/005000
lerobot-convert-dcp --checkpoint_dir=... --delete_dcp=true --push_to_hub=user/my-policy
lerobot-convert-dcp --repo_id=user/my-run
lerobot-convert-dcp --repo_id=user/my-run --checkpoint_step=5000 --push_to_hub=user/my-run
```

`--repo_id` downloads the checkpoint into `--output_dir` and then behaves exactly as if that
directory had been passed as `--checkpoint_dir`. Only runs that pushed with
`--save_checkpoint_to_hub` carry DCP shards on the Hub: a published model repo holds
safetensors only, so there is nothing there to convert.

`--push_to_hub` publishes the converted directory as a model repo, degrading gracefully: the
core artifacts (model.safetensors, config.json, processor files) always upload; the README
model card is enriched with training/dataset metadata only when `train_config.json` (and the
dataset it names) are reachable, with a WARNING naming exactly what was skipped otherwise.
DCP shard artifacts are never uploaded — published repos carry safetensors only.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from lerobot.configs import parser
from lerobot.distributed.checkpoint import dcp_to_safetensors
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.utils.hub import list_checkpoint_steps
from lerobot.utils.utils import init_logging


@dataclass
class ConvertDcpConfig:
    """CLI config for the offline DCP-to-safetensors checkpoint conversion."""

    # A local checkpoint step directory (containing pretrained_model/) or a pretrained_model
    # directory itself. Mutually exclusive with repo_id.
    checkpoint_dir: Path | None = None
    # A Hub model repo to read the checkpoint from, e.g. "user/my-run". Mutually exclusive
    # with checkpoint_dir. This is the source; --push_to_hub is the destination.
    repo_id: str | None = None
    # Repo revision (branch, tag, or commit sha) to read. Checkpoint pushes tag every commit
    # with its step, so a step identifier also works here. Requires repo_id.
    revision: str | None = None
    # Which checkpoints/<step>/ directory to convert; defaults to the highest step present.
    # Zero padding is optional — "5000" and "005000" both select step 5000. Requires repo_id.
    checkpoint_step: str | None = None
    # Where the downloaded checkpoint is written; defaults to
    # outputs/convert_dcp/<repo id with "/" replaced>. Requires repo_id.
    output_dir: Path | None = None
    # Remove the DCP shard directory after a successful conversion. With repo_id this removes
    # the downloaded copy only — the Hub repo keeps its shards.
    delete_dcp: bool = False
    # Publish the converted directory to this Hub repo id (e.g. "user/my-policy").
    push_to_hub: str | None = None
    private: bool | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or inert input rather than silently picking a source.

        Raises:
            ValueError: If neither or both of `checkpoint_dir` and `repo_id` are given, or if a
                Hub-only option is set while converting a local directory.
        """
        if (self.checkpoint_dir is None) == (self.repo_id is None):
            raise ValueError(
                "Pass exactly one source: --checkpoint_dir for a local checkpoint, or --repo_id "
                "for one on the Hub."
            )
        if self.repo_id is None:
            inert = [n for n in ("revision", "checkpoint_step", "output_dir") if getattr(self, n) is not None]
            if inert:
                raise ValueError(
                    f"{', '.join(f'--{n}' for n in inert)}: only meaningful with --repo_id. "
                    "--checkpoint_dir converts a directory in place."
                )


def _locate_pretrained_dir(checkpoint_dir: Path) -> Path:
    """Resolve the pretrained_model/ directory from a user-supplied checkpoint path.

    Args:
        checkpoint_dir (Path): A checkpoint step directory (containing `pretrained_model/`) or a
            `pretrained_model` directory itself.

    Returns:
        Path: The nested `pretrained_model/` directory when present, otherwise `checkpoint_dir`
        unchanged.
    """
    nested = checkpoint_dir / PRETRAINED_MODEL_DIR
    return nested if nested.is_dir() else checkpoint_dir


def _repo_join(*parts: str) -> str:
    """Join repo-relative path segments, dropping the empty ones that stand for the repo root."""
    return "/".join(p for p in parts if p)


def _default_output_dir(repo_id: str) -> Path:
    """Where a Hub checkpoint lands when `--output_dir` is not given.

    Args:
        repo_id (str): The source Hub model repo id.

    Returns:
        Path: `outputs/convert_dcp/<repo id with "/" replaced by "_">`.
    """
    return Path("outputs") / "convert_dcp" / repo_id.replace("/", "_")


def _select_checkpoint_prefix(files: list[str], repo_id: str, checkpoint_step: str | None) -> str:
    """Pick which directory of a Hub repo to convert, from its file listing.

    Training runs push to `checkpoints/<step>/` (see `push_checkpoint_to_hub`), so a step is
    selected there; a repo with no such tree is assumed to hold a checkpoint uploaded to its
    root by hand.

    Args:
        files (list[str]): Repo-relative file paths, as returned by `list_repo_files`.
        repo_id (str): The source repo id, for error messages.
        checkpoint_step (str | None): The requested step, zero padding optional. None selects
            the highest step present.

    Returns:
        str: The repo-relative directory to convert, or `""` for the repo root.

    Raises:
        ValueError: If `checkpoint_step` is not a step number.
        FileNotFoundError: If `checkpoint_step` names a step the repo does not have.
    """
    steps = list_checkpoint_steps(files)
    if checkpoint_step is not None:
        if not checkpoint_step.isdigit():
            raise ValueError(f"--checkpoint_step takes a step number, got '{checkpoint_step}'.")
        # Padding width follows the run's total step count, so compare numerically.
        match = next((s for s in steps if int(s) == int(checkpoint_step)), None)
        if match is None:
            available = ", ".join(sorted(steps, key=int)) or "none"
            raise FileNotFoundError(
                f"'{repo_id}' has no checkpoint at step {int(checkpoint_step)} (available: {available})."
            )
        return f"{CHECKPOINTS_DIR}/{match}"
    if steps:
        return f"{CHECKPOINTS_DIR}/{max(steps, key=int)}"
    return ""


def _fetch_hub_checkpoint(
    repo_id: str,
    output_dir: Path,
    *,
    revision: str | None = None,
    checkpoint_step: str | None = None,
) -> Path:
    """Download a DCP checkpoint from a Hub training repo, and return its local directory.

    Only the model shards are fetched. The step directory on the Hub also holds
    `training_state/`, whose optimizer shards outweigh the model itself and which conversion
    never reads, so it is left there.

    Args:
        repo_id (str): The Hub model repo holding the checkpoint.
        output_dir (Path): The local directory the checkpoint is downloaded into.
        revision (str | None): Repo revision (branch, tag, or commit sha) to read. Defaults to
            None (the default branch).
        checkpoint_step (str | None): Which `checkpoints/<step>/` directory to fetch. Defaults
            to None (the highest step present).

    Returns:
        Path: The downloaded checkpoint directory, ready to be passed to
        `_locate_pretrained_dir`.

    Raises:
        FileNotFoundError: If the selected directory holds no DCP shards.
    """
    from accelerate.utils.constants import FSDP_MODEL_NAME

    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    prefix = _select_checkpoint_prefix(files, repo_id, checkpoint_step)

    # Mirror _locate_pretrained_dir against the listing, so an unconvertible repo fails before
    # a single byte is downloaded: shards sit under <prefix>/pretrained_model/ for a checkpoint
    # pushed by a training run, or directly under <prefix> for a pretrained_model directory
    # uploaded on its own.
    shard_dir = f"{FSDP_MODEL_NAME}_0"
    nested = _repo_join(prefix, PRETRAINED_MODEL_DIR)
    for candidate in (nested, prefix):
        if any(f.startswith(f"{_repo_join(candidate, shard_dir)}/") for f in files):
            download_root = candidate
            break
    else:
        raise FileNotFoundError(
            f"No DCP shard directory ('{shard_dir}') under '{prefix or '<repo root>'}' in "
            f"'{repo_id}'. Shards reach the Hub only from a run trained with "
            "checkpoint_format=dcp (or safetensors_dcp) and --save_checkpoint_to_hub; a "
            "published model repo carries safetensors only."
        )

    logging.info(f"Downloading {_repo_join(repo_id, download_root)} -> {output_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=f"{download_root}/*" if download_root else None,
        local_dir=str(output_dir),
    )
    # snapshot_download keeps repo-relative paths, so the step directory reappears under
    # output_dir exactly as it is laid out on the Hub.
    return output_dir / prefix if prefix else output_dir


def _publish_converted(pretrained_dir: Path, repo_id: str, private: bool | None) -> None:
    """Best-effort publish of a converted checkpoint dir, degrading gracefully.

    The core artifacts (model.safetensors, config.json, processor files) always upload; the README
    model card gains training/dataset metadata only when `train_config.json` (and the dataset it
    names) are reachable, with a WARNING naming what was skipped otherwise. DCP shard artifacts are
    excluded from the upload.

    Args:
        pretrained_dir (Path): The converted `pretrained_model/` directory to upload.
        repo_id (str): Target Hub model repo id (e.g. "user/my-policy"); created if missing.
        private (bool | None): Repo visibility passed to `create_repo`; None keeps the Hub (or
            existing repo's) default.
    """
    from lerobot.common.train_utils import generate_model_card
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig

    train_cfg = None
    dataset_meta = None
    if (pretrained_dir / TRAIN_CONFIG_NAME).is_file():
        try:
            train_cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
        except Exception as e:  # noqa: BLE001 — degrade, never block the upload
            logging.warning(f"Could not parse {TRAIN_CONFIG_NAME} ({e}); README will lack training metadata.")
    else:
        logging.warning(f"{TRAIN_CONFIG_NAME} missing; README will lack training metadata.")
    if train_cfg is not None:
        try:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            dataset_meta = LeRobotDatasetMetadata(
                repo_id=train_cfg.dataset.repo_id,
                root=train_cfg.dataset.root,
                revision=train_cfg.dataset.revision,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(
                f"Dataset '{train_cfg.dataset.repo_id}' unreachable ({e}); README will lack dataset metadata."
            )
    try:
        model_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
        card = generate_model_card(model_cfg, cfg=train_cfg, dataset_meta=dataset_meta)
        card.save(str(pretrained_dir / "README.md"))
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Could not build the model card ({e}); publishing without README.")

    api = HfApi()
    repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(pretrained_dir),
        commit_message="Upload converted policy (DCP -> safetensors)",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        # The checkpoint keeps its DCP shard directory unless --delete_dcp was passed; the
        # allow list above admits neither `.distcp` shards nor their `.metadata` sidecar.
        ignore_patterns=["*.tmp", "*.log"],
    )
    logging.info(f"Model pushed to {commit_info.repo_url.url}")


@parser.wrap()
def convert_checkpoint(cfg: ConvertDcpConfig) -> Path:
    """Merge a checkpoint's DCP shards into `model.safetensors`, then optionally publish it.

    Args:
        cfg (ConvertDcpConfig): Conversion options — where the checkpoint comes from (a local
            directory or a Hub repo), whether to delete the DCP shards after a successful merge,
            and the optional Hub repo id (and visibility) to publish the converted directory to.

    Returns:
        Path: The path to the merged `model.safetensors` file.

    Raises:
        FileNotFoundError: If the checkpoint has no DCP shard directory, i.e. it was not saved
            with `checkpoint_format=dcp` (or `safetensors_dcp`).
    """
    from accelerate.utils.constants import FSDP_MODEL_NAME

    if cfg.repo_id is not None:
        checkpoint_dir = _fetch_hub_checkpoint(
            cfg.repo_id,
            cfg.output_dir or _default_output_dir(cfg.repo_id),
            revision=cfg.revision,
            checkpoint_step=cfg.checkpoint_step,
        )
    else:
        checkpoint_dir = cfg.checkpoint_dir
    pretrained_dir = _locate_pretrained_dir(checkpoint_dir)
    dcp_dir = pretrained_dir / f"{FSDP_MODEL_NAME}_0"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(
            f"No DCP shard directory at {dcp_dir}. Point --checkpoint_dir at a checkpoint "
            "saved with checkpoint_format=dcp (or safetensors_dcp)."
        )
    if cfg.delete_dcp and cfg.repo_id:
        logging.info(f"--delete_dcp removes the downloaded shards only; '{cfg.repo_id}' keeps its copy.")
    logging.info(f"Merging {dcp_dir} -> {pretrained_dir / 'model.safetensors'}")
    safetensors_path = dcp_to_safetensors(dcp_dir, pretrained_dir, delete_dcp=cfg.delete_dcp)
    if cfg.push_to_hub:
        _publish_converted(pretrained_dir, cfg.push_to_hub, cfg.private)
    return safetensors_path


def main() -> None:
    """`lerobot-convert-dcp` console entry point: set up logging and run the conversion."""
    init_logging()
    convert_checkpoint()


if __name__ == "__main__":
    main()
