import hashlib

import pytest

from isa.device_sweeps.checkpoints import download_entry


def manifest(digest):
    return {"repository": "example/device-models", "revision": "a" * 40,
            "checkpoints": [{"id": "selected", "status": "verified", "file": "model.pth",
                             "configuration_file": "config.json", "sha256": digest}]}


def test_download_is_pinned_and_verified(tmp_path, monkeypatch):
    config, weight = tmp_path / "config.json", tmp_path / "model.pth"
    config.write_text("{}")
    weight.write_bytes(b"example weights")
    calls = []

    def fake_download(filename, **kwargs):
        calls.append(kwargs)
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    data = manifest(hashlib.sha256(weight.read_bytes()).hexdigest())
    assert download_entry(data, "selected") == (config, weight)
    assert all(call["revision"] == "a" * 40 for call in calls)
    data["checkpoints"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_entry(data, "selected")


@pytest.mark.parametrize("revision", ["main", "release-tag", None, ""])
def test_download_rejects_mutable_or_missing_revision(revision):
    data = manifest("0" * 64)
    data["revision"] = revision
    with pytest.raises(ValueError, match="immutable"):
        download_entry(data, "selected")


def test_download_rejects_unverified_entry():
    data = manifest("0" * 64)
    data["checkpoints"][0]["status"] = "needs_review"
    with pytest.raises(ValueError, match="verification"):
        download_entry(data, "selected")
