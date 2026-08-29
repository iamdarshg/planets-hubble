from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataset.streaming import ManifestRecordError, StreamingDataset, StreamingSample


def _record(sample_id: str, *, split: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "arrays": {
            "flux": [1.0, 0.99, 1.0],
            "uncertainty": [0.01, 0.02, 0.01],
            "validity_mask": [True, True, False],
            "interpolation_mask": [False, True, False],
            "timestamps": [100.0, 100.1, 100.2],
            "wavelengths": [550.0, 650.0],
        },
        "metadata": {"time_system": "BJD_TDB", "exposure_duration_seconds": 30.0},
    }
    if split is not None:
        record["split"] = split
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_loads_one_sample_at_a_time_and_preserves_science_fields() -> None:
    loaded: list[str] = []

    def loader(record: dict[str, Any]) -> dict[str, Any]:
        loaded.append(record["sample_id"])
        return record

    def records() -> Any:
        for index in range(100):
            yield _record(f"sample-{index}")

    dataset = StreamingDataset(records(), loader=loader)
    iterator = iter(dataset)

    first = next(iterator)

    assert isinstance(first, StreamingSample)
    assert first.sample_id == "sample-0"
    assert loaded == ["sample-0"]
    assert first.arrays["flux"] == [1.0, 0.99, 1.0]
    assert first.uncertainty == [0.01, 0.02, 0.01]
    assert first.validity_mask == [True, True, False]
    assert first.interpolation_mask == [False, True, False]
    assert first.timestamps == [100.0, 100.1, 100.2]
    assert first.wavelengths == [550.0, 650.0]
    assert first.metadata["time_system"] == "BJD_TDB"


def test_jsonl_source_and_explicit_missingness_are_streamed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [_record("missingness")])

    sample = next(iter(StreamingDataset(manifest)))

    assert sample.arrays["flux"][2] == 1.0
    assert sample.uncertainty[1] == 0.02
    assert sample.validity_mask == [True, True, False]
    assert sample.interpolation_mask == [False, True, False]
    assert sample.metadata == {
        "time_system": "BJD_TDB",
        "exposure_duration_seconds": 30.0,
    }


def test_split_and_buffered_shuffle_are_deterministic_without_full_materialization() -> None:
    records = [_record(f"sample-{index}") for index in range(20)]
    kwargs = {"split": "all", "shuffle_buffer_size": 4, "seed": 17, "epoch": 3}

    first = [sample.sample_id for sample in StreamingDataset(records, **kwargs)]
    second = [sample.sample_id for sample in StreamingDataset(records, **kwargs)]

    assert first == second
    assert {sample_id for sample_id in first} == {
        f"sample-{index}" for index in range(20)
    }

    train_records = [_record(f"train-{index}", split="train") for index in range(3)]
    validation_records = [_record(f"validation-{index}", split="validation") for index in range(2)]
    split_dataset = StreamingDataset(
        train_records + validation_records,
        split="validation",
        shuffle_buffer_size=2,
    )

    assert [sample.sample_id for sample in split_dataset] == [
        "validation-0",
        "validation-1",
    ]


def test_shuffle_buffer_is_bounded_and_does_not_eagerly_load_samples() -> None:
    yielded_records = 0
    loaded: list[str] = []

    def records() -> Any:
        nonlocal yielded_records
        for index in range(1000):
            yielded_records += 1
            yield _record(f"sample-{index}")

    def loader(record: dict[str, Any]) -> dict[str, Any]:
        loaded.append(record["sample_id"])
        return record

    dataset = StreamingDataset(records(), loader=loader, shuffle_buffer_size=5, seed=1)
    first = next(iter(dataset))

    assert first.sample_id in {f"sample-{index}" for index in range(5)}
    assert len(loaded) == 1
    assert yielded_records <= 6
    assert dataset.shuffle_buffer_size == 5


def test_malformed_jsonl_record_reports_line_number(tmp_path: Path) -> None:
    manifest = tmp_path / "malformed.jsonl"
    manifest.write_text(
        json.dumps(_record("valid")) + "\n" + "{not-json}\n", encoding="utf-8"
    )

    iterator = iter(StreamingDataset(manifest))
    assert next(iterator).sample_id == "valid"

    with pytest.raises(ManifestRecordError, match=r"line 2.*invalid JSON"):
        next(iterator)


def test_record_without_sample_id_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "missing-id.jsonl"
    manifest.write_text(json.dumps({"arrays": {"flux": [1.0]}}) + "\n", encoding="utf-8")

    with pytest.raises(ManifestRecordError, match="sample_id"):
        next(iter(StreamingDataset(manifest)))
