"""Build bounded training manifests without generating or downloading payloads.

Synthetic records point at ``SyntheticConfig`` parameters and a sample index.
Real records index files already present below a caller-selected directory;
the files stay external to the repository and are never copied by this tool.

Example (synthetic-only, safe on a fresh checkout)::

    python examples/build_training_manifests.py \
      --synthetic-output data/manifests/synthetic-pretraining.jsonl \
      --summary-output data/manifests/synthetic-summary.json \
      --synthetic-records 8 --max-records 8 --max-bytes 65536
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_MAX_RECORDS = 10_000
DEFAULT_MAX_BYTES = 5 * 1024**3
PAYLOAD_SUFFIXES = (".fits", ".fit", ".fits.gz", ".npz")
REAL_ROLES = ("grpo", "sft", "positive_training")


class ManifestError(RuntimeError):
    """Raised when a manifest request cannot be satisfied within its contract."""


def _check_caps(max_records: int, max_bytes: int) -> None:
    if max_records < 1:
        raise ManifestError("max_records must be positive")
    if max_bytes < 1:
        raise ManifestError("max_bytes must be positive")


def _json_line(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_jsonl_atomic(
    output: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    max_records: int,
    max_bytes: int | None,
) -> tuple[int, int]:
    """Write records atomically, enforcing count and optional encoded-byte caps."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    count = 0
    total_bytes = 0
    try:
        with temporary.open("wb") as handle:
            for record in records:
                if count >= max_records:
                    raise ManifestError(f"record count exceeds max_records={max_records}")
                encoded = _json_line(record)
                if max_bytes is not None and total_bytes + len(encoded) > max_bytes:
                    raise ManifestError(f"JSONL size exceeds max_bytes={max_bytes}")
                handle.write(encoded)
                count += 1
                total_bytes += len(encoded)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count, total_bytes


def _split_for_sample(sample_index: int, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_index}:split".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def build_synthetic_manifest(
    output: Path,
    *,
    sample_count: int,
    seed: int = 0,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    validation_fraction: float = 0.2,
    config: Any | None = None,
) -> dict[str, Any]:
    """Write deterministic procedural references, never synthetic raster data."""

    _check_caps(max_records, max_bytes)
    if sample_count < 1:
        raise ManifestError("sample_count must be positive")
    if sample_count > max_records:
        raise ManifestError(f"sample_count exceeds max_records={max_records}")
    if not 0.0 <= validation_fraction <= 1.0:
        raise ManifestError("validation_fraction must be in [0, 1]")

    from synthetic import SyntheticConfig

    if config is None:
        config = SyntheticConfig(seed=seed)
    elif not isinstance(config, SyntheticConfig):
        raise ManifestError("config must be a SyntheticConfig")
    elif config.seed != seed:
        raise ManifestError("seed must match config.seed when config is supplied")

    config_dict = asdict(config)

    def records() -> Iterator[Mapping[str, Any]]:
        for sample_index in range(sample_count):
            split = _split_for_sample(sample_index, seed, validation_fraction)
            yield {
                "record_id": f"synthetic-{seed}-{sample_index}",
                "source": "synthetic",
                "payload_kind": "procedural_reference",
                "split": split,
                "role": "pretraining",
                "seed": seed,
                "sample_index": sample_index,
                "synthetic_config": config_dict,
                "labels": {
                    "kind": "counterfactual_preference",
                    "preferred": "injected",
                    "rejected": "null",
                    "event_type": config.event_type,
                    "latent_positive": True,
                },
                "labels_verified": True,
                "scientific_status": "synthetic_counterfactual_ground_truth",
                "provenance": {
                    "kind": "procedural_synthetic",
                    "generator": "synthetic.SyntheticConfig",
                    "payload_stored": False,
                },
            }

    count, encoded_bytes = _write_jsonl_atomic(
        output,
        records(),
        max_records=max_records,
        max_bytes=max_bytes,
    )
    split_counts = Counter(record["split"] for record in _synthetic_records(sample_count, seed, validation_fraction))
    return {
        "counts": {"records": count, **dict(sorted(split_counts.items()))},
        "bytes": encoded_bytes,
        "source_targets": [],
        "blocked_external_assets": [],
        "split_policy": {
            "method": "sha256(seed:sample_index:split)",
            "validation_fraction": validation_fraction,
            "roles_disjoint": True,
        },
    }


def _synthetic_records(sample_count: int, seed: int, validation_fraction: float) -> Iterator[dict[str, str]]:
    for sample_index in range(sample_count):
        yield {"split": _split_for_sample(sample_index, seed, validation_fraction)}


def _relative_path(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _first_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalise_role(metadata: Mapping[str, Any]) -> str:
    raw = _first_value(metadata, ("training_role", "dataset_role", "role"))
    if raw is None:
        return "unlabeled"
    role = str(raw).strip().lower().replace("-", "_")
    if role in REAL_ROLES:
        return role
    if role in {"unlabeled", "unknown", "holdout"}:
        return "unlabeled"
    raise ManifestError(f"unsupported real training role {raw!r}; declare metadata explicitly")


def _source_manifest_path(path: Path, source_root: Path) -> str:
    return _relative_path(path, source_root)


def build_real_manifest(
    source_root: Path,
    output: Path,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    holdout_targets: Iterable[str] = (),
    holdout_lineages: Iterable[str] = (),
) -> dict[str, Any]:
    """Index local FITS/NPZ assets and manifests without downloading or copying."""

    _check_caps(max_records, max_bytes)
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ManifestError(f"source directory does not exist: {source_root}")

    holdout_target_keys = {str(value).strip().casefold() for value in holdout_targets}
    holdout_lineage_keys = {str(value).strip().casefold() for value in holdout_lineages}
    blocked: set[str] = set()
    seen: dict[str, str] = {}
    source_targets: set[str] = set()
    indexed_bytes = 0
    metadata_entries: list[tuple[Path, dict[str, Any], Path]] = []
    referenced_paths: set[Path] = set()

    manifest_documents: list[tuple[bool, Path, Mapping[str, Any]]] = []
    for manifest_path in sorted(source_root.rglob("*.json")):
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping) or not isinstance(document.get("records"), list):
            continue
        is_prepared = any(
            isinstance(item, Mapping) and isinstance(item.get("record"), Mapping)
            for item in document["records"]
        )
        manifest_documents.append((is_prepared, manifest_path, document))

    for _, manifest_path, document in sorted(manifest_documents, key=lambda item: (not item[0], item[1].as_posix())):
        for item in document["records"]:
            if not isinstance(item, Mapping):
                continue
            nested = item.get("record")
            metadata = dict(nested) if isinstance(nested, Mapping) else dict(item)
            for key in ("target", "target_id", "target_name", "training_role", "dataset_role", "role"):
                if key not in metadata and document.get(key) is not None:
                    metadata[key] = document[key]
            reference = _first_value(item, ("array_file", "path", "file", "data_path", "product_id"))
            if reference is None:
                continue
            referenced = Path(str(reference))
            if not referenced.is_absolute():
                referenced = manifest_path.parent / referenced
            try:
                resolved = referenced.resolve()
                relative = _relative_path(resolved, source_root)
            except ValueError:
                blocked.add(f"{_source_manifest_path(manifest_path, source_root)}:{reference}")
                continue
            referenced_paths.add(resolved)
            if not resolved.is_file() or not resolved.name.lower().endswith(PAYLOAD_SUFFIXES):
                blocked.add(relative)
                continue
            metadata_entries.append((resolved, metadata, manifest_path))

    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.resolve() not in referenced_paths and path.name.lower().endswith(PAYLOAD_SUFFIXES):
            metadata_entries.append((path.resolve(), {}, path.resolve()))

    def records() -> Iterator[Mapping[str, Any]]:
        nonlocal indexed_bytes
        for path, metadata, manifest_path in metadata_entries:
            relative = _relative_path(path, source_root)
            if relative in seen:
                previous_role = seen[relative]
                current_role = _normalise_role(metadata)
                if previous_role != current_role and {previous_role, current_role} <= set(REAL_ROLES):
                    raise ManifestError(f"conflicting real training roles for {relative}")
                continue
            size = path.stat().st_size
            if indexed_bytes + size > max_bytes:
                raise ManifestError(f"real payload bytes exceed max_bytes={max_bytes}")
            file_size, sha256 = _hash_file(path)
            if file_size != size:
                raise ManifestError(f"file changed while hashing: {relative}")
            record_id = "real-" + hashlib.sha256(
                f"{relative}:{sha256}".encode("utf-8")
            ).hexdigest()[:24]
            target_value = _first_value(metadata, ("target", "target_id", "target_name"))
            target = None if target_value is None else str(target_value)
            lineage_value = _first_value(
                metadata,
                ("observation_lineage", "lineage_id", "parent_observation_id", "observation_id", "obsid", "visit_id"),
            )
            lineage = None if lineage_value is None else str(lineage_value)
            role = _normalise_role(metadata)
            target_holdout = target is not None and target.strip().casefold() in holdout_target_keys
            lineage_holdout = lineage is not None and lineage.strip().casefold() in holdout_lineage_keys
            if target_holdout or lineage_holdout:
                role = "holdout"
            labels_verified = metadata.get("labels_verified") is True
            scientific_status = _first_value(metadata, ("scientific_status",))
            if scientific_status is None:
                scientific_status = (
                    "unlabeled_real_observation"
                    if role == "unlabeled"
                    else "real_observation_role_declared_without_truth_inference"
                )
            record: dict[str, Any] = {
                "record_id": record_id,
                "source": "real_hubble",
                "path": relative,
                "size_bytes": file_size,
                "sha256": sha256,
                "target": target,
                "observation_lineage": lineage,
                "role": role,
                "training_eligible": role in REAL_ROLES and role != "holdout",
                "labels_verified": labels_verified,
                "scientific_status": str(scientific_status),
                "provenance": {
                    "kind": "external_local_data",
                    "source_manifest": _source_manifest_path(manifest_path, source_root)
                    if manifest_path != path
                    else None,
                    "payload_stored_in_repository": False,
                },
            }
            if metadata.get("label_metadata") is not None:
                record["label_metadata"] = metadata["label_metadata"]
            seen[relative] = role
            indexed_bytes += file_size
            if target is not None:
                source_targets.add(target)
            yield record

    count, _ = _write_jsonl_atomic(
        output,
        records(),
        max_records=max_records,
        max_bytes=None,
    )
    role_counts = Counter(seen.values())
    counts = {"records": count, **{role: role_counts.get(role, 0) for role in (*REAL_ROLES, "holdout", "unlabeled")}}
    return {
        "counts": counts,
        "bytes": indexed_bytes,
        "source_targets": sorted(source_targets),
        "blocked_external_assets": sorted(blocked),
        "split_policy": {
            "method": "explicit metadata role plus target/lineage holdout exclusion",
            "roles": list(REAL_ROLES),
            "holdout_targets": sorted(holdout_target_keys),
            "holdout_lineages": sorted(holdout_lineage_keys),
            "roles_disjoint": True,
        },
    }


def build_dataset_manifests(
    *,
    synthetic_output: Path,
    summary_output: Path,
    synthetic_records: int,
    real_root: Path | None = None,
    real_output: Path | None = None,
    seed: int = 0,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    validation_fraction: float = 0.2,
    holdout_targets: Iterable[str] = (),
    holdout_lineages: Iterable[str] = (),
) -> dict[str, Any]:
    """Build selected manifests and write one summary JSON."""

    if real_root is not None:
        if real_output is None:
            raise ManifestError("real_output is required when real_root is supplied")
        if not real_root.resolve().is_dir():
            raise ManifestError(f"source directory does not exist: {real_root.resolve()}")

    synthetic_summary = build_synthetic_manifest(
        synthetic_output,
        sample_count=synthetic_records,
        seed=seed,
        max_records=max_records,
        max_bytes=max_bytes,
        validation_fraction=validation_fraction,
    )
    real_summary = None
    if real_root is not None:
        real_summary = build_real_manifest(
            real_root,
            real_output,
            max_records=max_records,
            max_bytes=max_bytes,
            holdout_targets=holdout_targets,
            holdout_lineages=holdout_lineages,
        )
    summaries = [synthetic_summary] + ([real_summary] if real_summary is not None else [])
    summary = {
        "schema_version": 1,
        "counts": {
            "synthetic_records": synthetic_summary["counts"]["records"],
            "real_records": 0 if real_summary is None else real_summary["counts"]["records"],
        },
        "bytes": {
            "synthetic_manifest": synthetic_summary["bytes"],
            "real_indexed_payloads": 0 if real_summary is None else real_summary["bytes"],
        },
        "source_targets": sorted({target for item in summaries for target in item["source_targets"]}),
        "split_policy": {
            "synthetic": synthetic_summary["split_policy"],
            "real": None if real_summary is None else real_summary["split_policy"],
        },
        "caps": {"max_records": max_records, "max_bytes": max_bytes},
        "blocked_external_assets": sorted(
            {asset for item in summaries for asset in item["blocked_external_assets"]}
        ),
        "synthetic": synthetic_summary,
        "real": real_summary,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--synthetic-records", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--real-output", type=Path)
    parser.add_argument("--holdout-target", action="append", default=[])
    parser.add_argument("--holdout-lineage", action="append", default=[])
    args = parser.parse_args()
    try:
        build_dataset_manifests(
            synthetic_output=args.synthetic_output,
            summary_output=args.summary_output,
            synthetic_records=args.synthetic_records,
            real_root=args.real_root,
            real_output=args.real_output,
            seed=args.seed,
            max_records=args.max_records,
            max_bytes=args.max_bytes,
            validation_fraction=args.validation_fraction,
            holdout_targets=args.holdout_target,
            holdout_lineages=args.holdout_lineage,
        )
    except ManifestError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
