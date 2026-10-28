"""Cleanup must remove only this benchmark's verified temporary model files."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/step_speed"))
from model_storage import cleanup_owned, create_owned


def test_private_model_cleanup_preserves_external_symlink_target(tmp_path):
    campaign = tmp_path / "results"
    external = tmp_path / "shared-model"
    external.mkdir()
    (external / "weights").write_bytes(b"shared")
    owned = create_owned(campaign, base=tmp_path)
    (owned / "weights").write_bytes(b"private-model")
    (owned / "shared-link").symlink_to(external, target_is_directory=True)
    result = cleanup_owned(owned, campaign, base=tmp_path)
    assert result["removed"] and not owned.exists()
    assert result["removed_file_bytes"] >= len(b"private-model")
    assert (external / "weights").read_bytes() == b"shared"


def test_cleanup_rejects_another_campaign(tmp_path):
    owned = create_owned(tmp_path / "first", base=tmp_path)
    with pytest.raises(ValueError):
        cleanup_owned(owned, tmp_path / "second", base=tmp_path)
    assert owned.exists()


def test_cleanup_rejects_symlink_root(tmp_path):
    campaign = tmp_path / "results"
    owned = create_owned(campaign, base=tmp_path)
    link = tmp_path / "lora-step-speed-8b-20261028-link"
    link.symlink_to(owned, target_is_directory=True)
    with pytest.raises(ValueError):
        cleanup_owned(link, campaign, base=tmp_path)
    assert owned.exists()


def test_cleanup_rejects_nonprivate_directory(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    with pytest.raises(ValueError):
        cleanup_owned(directory, tmp_path / "results", base=tmp_path)
    assert directory.exists()
