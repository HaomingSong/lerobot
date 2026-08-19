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
"""lerobot-convert-dcp: locating, converting, and graceful-degradation publishing."""

import logging
import shutil
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

import lerobot.distributed.checkpoint as dist_checkpoint
import lerobot.scripts.lerobot_convert_dcp as convert_mod
from lerobot.scripts.lerobot_convert_dcp import (
    ConvertDcpConfig,
    _default_output_dir,
    _locate_pretrained_dir,
    _publish_converted,
    _select_checkpoint_prefix,
    convert_checkpoint,
)
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR


@pytest.fixture
def fake_merge(monkeypatch):
    """Stand in for accelerate.utils.merge_fsdp_weights: writes a marker safetensors file."""
    import accelerate.utils

    def merge(checkpoint_dir, output_path, safe_serialization=True, remove_checkpoint_dir=False):
        assert isinstance(checkpoint_dir, str) and isinstance(output_path, str)  # str, not Path
        (Path(output_path) / "model.safetensors").write_bytes(b"merged")
        # Mirror accelerate: the shard directory is removed by the merge itself, when asked.
        if remove_checkpoint_dir:
            shutil.rmtree(checkpoint_dir)

    monkeypatch.setattr(accelerate.utils, "merge_fsdp_weights", merge)


def make_dcp_checkpoint(tmp_path: Path) -> Path:
    pretrained = tmp_path / PRETRAINED_MODEL_DIR
    dcp_dir = pretrained / "pytorch_model_fsdp_0"
    dcp_dir.mkdir(parents=True)
    (dcp_dir / "__0_0.distcp").write_bytes(b"shard")
    (pretrained / "config.json").write_text("{}")
    return tmp_path


def hub_files(*steps: str) -> list[str]:
    """File listing of a training repo that pushed DCP checkpoints for the given steps.

    With no steps, the checkpoint sits at the repo root instead — the layout you get by
    uploading a checkpoint directory by hand rather than with --save_checkpoint_to_hub.
    """
    files = ["README.md", "config.json", "model.safetensors"]
    for prefix in [f"checkpoints/{s}/" for s in steps] or [""]:
        files += [
            f"{prefix}{PRETRAINED_MODEL_DIR}/pytorch_model_fsdp_0/__0_0.distcp",
            f"{prefix}{PRETRAINED_MODEL_DIR}/pytorch_model_fsdp_0/.metadata",
            f"{prefix}{PRETRAINED_MODEL_DIR}/config.json",
            f"{prefix}{TRAINING_STATE_DIR}/optimizer_0/__0_0.distcp",
            f"{prefix}{TRAINING_STATE_DIR}/training_step.json",
        ]
    return files


@pytest.fixture
def fake_hub(monkeypatch):
    """Stand in for the Hub: `files` drives the listing, and downloads materialize from it.

    `snapshot_download` filters with fnmatch, where `*` spans `/` — mirrored here so a test can
    assert on which parts of a repo an allow-pattern actually pulls.
    """
    hub = SimpleNamespace(files=[], calls=[], downloaded=None)

    class FakeApi:
        def list_repo_files(self, *, repo_id, repo_type, revision=None, token=None):
            hub.calls.append(("list", repo_id, revision))
            return hub.files

    def fake_snapshot_download(*, repo_id, repo_type, revision, allow_patterns, local_dir):
        hub.downloaded = SimpleNamespace(
            repo_id=repo_id, revision=revision, allow_patterns=allow_patterns, local_dir=local_dir
        )
        for f in hub.files:
            if allow_patterns is not None and not fnmatch(f, allow_patterns):
                continue
            dest = Path(local_dir) / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
        return local_dir

    monkeypatch.setattr(convert_mod, "HfApi", FakeApi)
    monkeypatch.setattr(convert_mod, "snapshot_download", fake_snapshot_download)
    return hub


class TestConvert:
    def test_locate_accepts_step_dir_or_pretrained_dir(self, tmp_path):
        step_dir = make_dcp_checkpoint(tmp_path)
        pretrained = step_dir / PRETRAINED_MODEL_DIR
        assert _locate_pretrained_dir(step_dir) == pretrained
        assert _locate_pretrained_dir(pretrained) == pretrained

    def test_convert_keeps_dcp_by_default(self, tmp_path, fake_merge):
        step_dir = make_dcp_checkpoint(tmp_path)
        out = convert_checkpoint(ConvertDcpConfig(checkpoint_dir=step_dir))
        assert out.read_bytes() == b"merged"
        assert (step_dir / PRETRAINED_MODEL_DIR / "pytorch_model_fsdp_0").is_dir()

    def test_convert_delete_dcp(self, tmp_path, fake_merge):
        step_dir = make_dcp_checkpoint(tmp_path)
        convert_checkpoint(ConvertDcpConfig(checkpoint_dir=step_dir, delete_dcp=True))
        assert not (step_dir / PRETRAINED_MODEL_DIR / "pytorch_model_fsdp_0").exists()

    def test_missing_shards_error_names_the_format(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="checkpoint_format=dcp"):
            convert_checkpoint(ConvertDcpConfig(checkpoint_dir=tmp_path))


class TestPublishGracefulDegradation:
    def _mock_api(self, monkeypatch):
        calls = {}

        class FakeApi:
            def create_repo(self, repo_id, private=None, exist_ok=False):
                return SimpleNamespace(repo_id=repo_id)

            def upload_folder(self, *, repo_id, folder_path, allow_patterns, **kwargs):
                calls["repo_id"] = repo_id
                calls["files"] = sorted(p.name for p in Path(folder_path).iterdir())
                calls["allow_patterns"] = allow_patterns
                return SimpleNamespace(repo_url=SimpleNamespace(url=f"https://huggingface.co/{repo_id}"))

        import lerobot.scripts.lerobot_convert_dcp as mod

        monkeypatch.setattr(mod, "HfApi", FakeApi)
        return calls

    def test_missing_train_config_warns_and_uploads_core(self, tmp_path, monkeypatch, caplog):
        calls = self._mock_api(monkeypatch)
        pretrained = make_dcp_checkpoint(tmp_path) / PRETRAINED_MODEL_DIR
        (pretrained / "model.safetensors").write_bytes(b"w")
        with caplog.at_level(logging.WARNING):
            _publish_converted(pretrained, "user/converted", private=None)
        assert any("train_config.json missing" in m for m in caplog.messages)
        assert "model.safetensors" in calls["files"]
        # The DCP shard directory is still on disk (--delete_dcp defaults to False) but the
        # allow list admits neither `.distcp` shards nor their `.metadata` sidecar.
        assert set(calls["allow_patterns"]) == {"*.safetensors", "*.json", "*.yaml", "*.md"}
        # config.json is not parseable as a policy config here -> card skipped with a warning
        assert any("model card" in m for m in caplog.messages)

    def test_dcp_to_safetensors_passes_str_paths(self, tmp_path, fake_merge):
        """accelerate 1.14's DCP helpers do string containment checks."""
        dcp_dir = tmp_path / "pytorch_model_fsdp_0"
        dcp_dir.mkdir()
        out = dist_checkpoint.dcp_to_safetensors(dcp_dir, tmp_path, delete_dcp=True)
        assert out == tmp_path / "model.safetensors"
        assert not dcp_dir.exists()


class TestSourceValidation:
    """Exactly one source, and no Hub-only option that would sit there doing nothing."""

    def test_a_source_is_required(self):
        with pytest.raises(ValueError, match="exactly one source"):
            ConvertDcpConfig()

    def test_two_sources_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one source"):
            ConvertDcpConfig(checkpoint_dir=tmp_path, repo_id="user/run")

    @pytest.mark.parametrize(
        ("option", "value"),
        [("revision", "main"), ("checkpoint_step", "5000"), ("output_dir", Path("out"))],
    )
    def test_hub_options_need_a_hub_source(self, tmp_path, option, value):
        with pytest.raises(ValueError, match=f"--{option}: only meaningful with --repo_id"):
            ConvertDcpConfig(checkpoint_dir=tmp_path, **{option: value})


class TestCheckpointSelection:
    def test_defaults_to_the_highest_step(self):
        assert _select_checkpoint_prefix(hub_files("005000", "010000"), "user/run", None) == (
            "checkpoints/010000"
        )

    def test_explicit_step_ignores_zero_padding(self):
        files = hub_files("005000", "010000")
        assert _select_checkpoint_prefix(files, "user/run", "5000") == "checkpoints/005000"
        assert _select_checkpoint_prefix(files, "user/run", "005000") == "checkpoints/005000"

    def test_unknown_step_names_what_is_available(self):
        with pytest.raises(FileNotFoundError, match="step 7000 .available: 005000, 010000."):
            _select_checkpoint_prefix(hub_files("005000", "010000"), "user/run", "7000")

    def test_step_must_be_a_number(self):
        with pytest.raises(ValueError, match="takes a step number"):
            _select_checkpoint_prefix(hub_files("005000"), "user/run", "last")

    def test_repo_without_a_checkpoints_tree_falls_back_to_the_root(self):
        assert _select_checkpoint_prefix(hub_files(), "user/run", None) == ""

    def test_default_output_dir_is_derived_from_the_repo_id(self):
        assert _default_output_dir("user/my-run") == Path("outputs/convert_dcp/user_my-run")


class TestConvertFromHub:
    def test_converts_the_latest_checkpoint(self, tmp_path, fake_hub, fake_merge):
        fake_hub.files = hub_files("005000", "010000")
        out = convert_checkpoint(ConvertDcpConfig(repo_id="user/run", output_dir=tmp_path))
        assert out == tmp_path / "checkpoints/010000" / PRETRAINED_MODEL_DIR / "model.safetensors"
        assert out.read_bytes() == b"merged"

    def test_training_state_is_left_on_the_hub(self, tmp_path, fake_hub, fake_merge):
        """Optimizer shards outweigh the model and conversion never reads them."""
        fake_hub.files = hub_files("005000")
        convert_checkpoint(ConvertDcpConfig(repo_id="user/run", output_dir=tmp_path))
        assert fake_hub.downloaded.allow_patterns == f"checkpoints/005000/{PRETRAINED_MODEL_DIR}/*"
        assert not (tmp_path / "checkpoints/005000" / TRAINING_STATE_DIR).exists()
        assert not (tmp_path / "README.md").exists()

    def test_step_and_revision_reach_the_hub_calls(self, tmp_path, fake_hub, fake_merge):
        fake_hub.files = hub_files("005000", "010000")
        convert_checkpoint(
            ConvertDcpConfig(
                repo_id="user/run", output_dir=tmp_path, revision="refs/pr/1", checkpoint_step="5000"
            )
        )
        assert fake_hub.calls == [("list", "user/run", "refs/pr/1")]
        assert fake_hub.downloaded.revision == "refs/pr/1"
        assert fake_hub.downloaded.allow_patterns == f"checkpoints/005000/{PRETRAINED_MODEL_DIR}/*"

    def test_checkpoint_uploaded_to_the_repo_root(self, tmp_path, fake_hub, fake_merge):
        fake_hub.files = hub_files()
        out = convert_checkpoint(ConvertDcpConfig(repo_id="user/run", output_dir=tmp_path))
        assert out == tmp_path / PRETRAINED_MODEL_DIR / "model.safetensors"
        assert fake_hub.downloaded.allow_patterns == f"{PRETRAINED_MODEL_DIR}/*"
        assert not (tmp_path / TRAINING_STATE_DIR).exists()

    def test_repo_without_shards_fails_before_downloading(self, tmp_path, fake_hub):
        fake_hub.files = ["README.md", "config.json", "model.safetensors"]
        with pytest.raises(FileNotFoundError, match="carries safetensors only"):
            convert_checkpoint(ConvertDcpConfig(repo_id="user/published", output_dir=tmp_path))
        assert fake_hub.downloaded is None

    def test_delete_dcp_says_it_spares_the_hub_copy(self, tmp_path, fake_hub, fake_merge, caplog):
        fake_hub.files = hub_files("005000")
        with caplog.at_level(logging.INFO):
            convert_checkpoint(ConvertDcpConfig(repo_id="user/run", output_dir=tmp_path, delete_dcp=True))
        assert any("keeps its copy" in m for m in caplog.messages)
        assert not (tmp_path / "checkpoints/005000" / PRETRAINED_MODEL_DIR / "pytorch_model_fsdp_0").exists()
