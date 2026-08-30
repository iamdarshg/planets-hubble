from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "build_training_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_training_manifests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_payload(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_synthetic_manifest_is_deterministic_bounded_and_payload_free(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    MODULE.build_synthetic_manifest(
        first,
        sample_count=20,
        seed=19,
        max_records=20,
        max_bytes=100_000,
        validation_fraction=0.25,
    )
    MODULE.build_synthetic_manifest(
        second,
        sample_count=20,
        seed=19,
        max_records=20,
        max_bytes=100_000,
        validation_fraction=0.25,
    )

    assert first.read_bytes() == second.read_bytes()
    records = _read_jsonl(first)
    assert len(records) == 20
    assert {record["split"] for record in records} == {"train", "validation"}
    for record in records:
        assert record["source"] == "synthetic"
        assert record["payload_kind"] == "procedural_reference"
        assert record["seed"] == 19
        assert isinstance(record["sample_index"], int)
        assert record["labels_verified"] is True
        assert record["scientific_status"] == "synthetic_counterfactual_ground_truth"
        assert record["labels"]["preferred"] == "injected"
        assert record["labels"]["rejected"] == "null"
        assert "raster" not in record
        assert "payload" not in record


def test_synthetic_manifest_fails_before_replacing_output_when_cap_is_too_small(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic.jsonl"
    output.write_text("old\n", encoding="utf-8")

    with pytest.raises(MODULE.ManifestError, match="max_bytes"):
        MODULE.build_synthetic_manifest(
            output,
            sample_count=2,
            seed=3,
            max_records=2,
            max_bytes=1,
        )

    assert output.read_text(encoding="utf-8") == "old\n"


def test_real_manifest_prefers_prepared_records_and_keeps_roles_honest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real"
    prepared = source / "prepared-target" / "manifest.json"
    holdout = source / "held-target" / "manifest.json"

    grpo_path = prepared.parent / "grpo.npz"
    sft_path = prepared.parent / "sft.npz"
    positive_path = prepared.parent / "positive.npz"
    held_path = holdout.parent / "held.npz"
    filename_named_like_truth = source / "confirmed-exoplanet.npz"
    _write_payload(grpo_path, b"grpo")
    _write_payload(sft_path, b"sft")
    _write_payload(positive_path, b"positive")
    _write_payload(held_path, b"held")
    _write_payload(filename_named_like_truth, b"unknown")

    prepared.write_text(
        json.dumps(
            {
                "target": "Training Star",
                "records": [
                    {
                        "record": {
                            "observation_id": "obs-grpo",
                            "training_role": "grpo",
                            "labels_verified": False,
                            "scientific_status": "real_observation_unlabeled",
                        },
                        "array_file": "grpo.npz",
                    },
                    {
                        "record": {
                            "observation_id": "obs-sft",
                            "training_role": "sft",
                            "labels_verified": True,
                            "scientific_status": "artifact_label_from_metadata",
                            "label_metadata": {"kind": "artifact", "source": "fixture"},
                        },
                        "array_file": "sft.npz",
                    },
                    {
                        "record": {
                            "observation_id": "obs-positive",
                            "training_role": "positive_training",
                            "labels_verified": True,
                            "scientific_status": "published_positive_metadata",
                            "label_metadata": {"kind": "positive", "source": "fixture"},
                        },
                        "array_file": "positive.npz",
                    },
                ],
                "source": "external_fixture",
            }
        ),
        encoding="utf-8",
    )
    holdout.write_text(
        json.dumps(
            {
                "target": "Held Star",
                "records": [
                    {
                        "record": {
                            "observation_id": "obs-held",
                            "training_role": "positive_training",
                            "labels_verified": True,
                            "scientific_status": "published_positive_metadata",
                        },
                        "array_file": "held.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "real.jsonl"
    summary = MODULE.build_real_manifest(
        source,
        output,
        max_records=10,
        max_bytes=1_000,
        holdout_targets={"Held Star"},
    )

    records = _read_jsonl(output)
    by_path = {record["path"]: record for record in records}
    assert set(by_path) == {
        "prepared-target/grpo.npz",
        "prepared-target/sft.npz",
        "prepared-target/positive.npz",
        "held-target/held.npz",
        "confirmed-exoplanet.npz",
    }
    assert by_path["prepared-target/grpo.npz"]["role"] == "grpo"
    assert by_path["prepared-target/sft.npz"]["role"] == "sft"
    assert by_path["prepared-target/positive.npz"]["role"] == "positive_training"
    assert by_path["held-target/held.npz"]["role"] == "holdout"
    assert by_path["held-target/held.npz"]["training_eligible"] is False
    assert by_path["confirmed-exoplanet.npz"]["role"] == "unlabeled"
    assert by_path["confirmed-exoplanet.npz"]["labels_verified"] is False
    assert by_path["confirmed-exoplanet.npz"]["scientific_status"] == "unlabeled_real_observation"
    assert all(record["provenance"]["kind"] == "external_local_data" for record in records)
    assert all(not Path(record["path"]).is_absolute() for record in records)
    assert by_path["prepared-target/grpo.npz"]["sha256"] == hashlib.sha256(b"grpo").hexdigest()
    assert summary["counts"]["grpo"] == 1
    assert summary["counts"]["sft"] == 1
    assert summary["counts"]["positive_training"] == 1
    assert summary["counts"]["holdout"] == 1
    assert summary["counts"]["unlabeled"] == 1
    assert summary["source_targets"] == ["Held Star", "Training Star"]


def test_real_manifest_missing_source_and_cap_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(MODULE.ManifestError, match="source directory"):
        MODULE.build_real_manifest(missing, tmp_path / "real.jsonl", max_records=2, max_bytes=100)

    source = tmp_path / "real"
    payload = source / "target" / "sample.npz"
    _write_payload(payload, b"0123456789")
    output = tmp_path / "real.jsonl"
    with pytest.raises(MODULE.ManifestError, match="max_bytes"):
        MODULE.build_real_manifest(source, output, max_records=2, max_bytes=5)
    assert not output.exists()


def test_real_manifest_reports_blocked_external_manifest_assets(tmp_path: Path) -> None:
    source = tmp_path / "real"
    manifest = source / "target" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "target": "Remote Star",
                "records": [
                    {
                        "observation_id": "remote-1",
                        "product_id": "remote.fits",
                        "product_uri": "mast:HST/product/remote.fits",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "real.jsonl"
    summary = MODULE.build_real_manifest(source, output, max_records=2, max_bytes=100)

    assert _read_jsonl(output) == []
    assert summary["counts"]["records"] == 0
    assert summary["blocked_external_assets"] == ["target/remote.fits"]
