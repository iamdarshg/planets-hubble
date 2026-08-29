"""Bounded, lazy loading for normalized observation manifests.

The streaming dataset deliberately has no ``__len__`` and never caches the
manifest.  A JSONL path is reopened for each iteration; an iterable source is
consumed as provided.  Samples are loaded only after a manifest record is
selected, so a bounded shuffle buffer contains records, not image arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ManifestRecordError(ValueError):
    """Raised when a manifest record is malformed or cannot be loaded."""


@dataclass(frozen=True)
class StreamingSample:
    """One normalized sample and its temporal/spectral science metadata.

    Values are intentionally typed as ``Any``: callers may use Python
    sequences, NumPy arrays, or read-only ``numpy.memmap`` instances supplied
    by the default loader or an injected loader.  No conversion or copying is
    performed here.
    """

    sample_id: str
    arrays: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Any = None
    validity_mask: Any = None
    interpolation_mask: Any = None
    timestamps: Any = None
    wavelengths: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_record: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def values(self) -> Any:
        """Return the primary values array under the common manifest aliases."""

        return _first_present(self.arrays, "values", "flux", "data")


SampleLoader = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ManifestSource = Path | str | Iterable[Mapping[str, Any]]


class StreamingDataset:
    """Iterate over manifest records with bounded memory use.

    ``manifest`` may be a JSONL path or a reusable/one-shot iterable of
    mappings.  The default loader supports inline ``arrays`` and paths to
    ``.npy``/``.npz`` files.  For FITS, Zarr, remote objects, or custom array
    stores, pass a ``loader`` that loads exactly one record at a time.

    Splits are assigned from an explicit record ``split`` field when present.
    Otherwise a stable hash of ``seed`` and ``sample_id`` assigns the record to
    train/validation/test.  This avoids a dataset-wide index or random state.
    """

    _SPLITS = {"all", "train", "validation", "val", "test"}

    def __init__(
        self,
        manifest: ManifestSource,
        *,
        loader: SampleLoader | None = None,
        split: str = "all",
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 0,
        split_seed: int | None = None,
        epoch: int = 0,
        shuffle_buffer_size: int = 0,
        max_samples: int | None = None,
    ) -> None:
        normalized_split = str(split).lower()
        if normalized_split not in self._SPLITS:
            raise ValueError("split must be one of all, train, validation, or test")
        if len(split_ratios) != 3 or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in split_ratios
        ):
            raise ValueError("split_ratios must contain three finite non-negative values")
        if not math.isclose(sum(split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("split_ratios must sum to 1")
        if not isinstance(shuffle_buffer_size, int) or shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be a non-negative integer")
        if max_samples is not None and (
            not isinstance(max_samples, int) or max_samples < 0
        ):
            raise ValueError("max_samples must be None or a non-negative integer")

        self.manifest = manifest
        self.loader = loader
        self.split = "validation" if normalized_split == "val" else normalized_split
        self.split_ratios = tuple(float(value) for value in split_ratios)
        self.seed = int(seed)
        self.split_seed = self.seed if split_seed is None else int(split_seed)
        self.epoch = int(epoch)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = max_samples
        self._manifest_directory = (
            Path(manifest).expanduser().resolve().parent
            if isinstance(manifest, (str, Path))
            else Path.cwd()
        )

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs: Any) -> "StreamingDataset":
        """Construct a dataset whose source is a JSONL manifest path."""

        return cls(path, **kwargs)

    def __iter__(self) -> Iterator[StreamingSample]:
        records = self._selected_records()
        if self.shuffle_buffer_size:
            records = self._shuffle_records(records)

        yielded = 0
        for record in records:
            if self.max_samples is not None and yielded >= self.max_samples:
                return
            yield self._load_sample(record)
            yielded += 1

    def _selected_records(self) -> Iterator[Mapping[str, Any]]:
        for line_number, raw_record in self._records():
            record = _validate_record(raw_record, line_number)
            if self._belongs_to_split(record):
                yield record

    def _records(self) -> Iterator[tuple[int | None, Mapping[str, Any]]]:
        if isinstance(self.manifest, (str, Path)):
            path = Path(self.manifest).expanduser()
            try:
                handle = path.open("r", encoding="utf-8")
            except OSError as exc:
                raise ManifestRecordError(f"cannot open manifest {path}: {exc}") from exc
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ManifestRecordError(
                            f"manifest line {line_number} contains invalid JSON: {exc.msg}"
                        ) from exc
                    yield line_number, record
            return

        if isinstance(self.manifest, Mapping):
            yield None, self.manifest
            return

        for record in self.manifest:
            yield None, record

    def _belongs_to_split(self, record: Mapping[str, Any]) -> bool:
        if self.split == "all":
            return True
        explicit = record.get("split")
        if explicit is not None:
            return str(explicit).lower() == self.split

        sample_id = str(record["sample_id"])
        digest = hashlib.blake2b(
            f"{self.split_seed}:{sample_id}".encode("utf-8"), digest_size=8
        ).digest()
        fraction = int.from_bytes(digest, "big") / float(2**64)
        train_end = self.split_ratios[0]
        validation_end = train_end + self.split_ratios[1]
        assigned = (
            "train"
            if fraction < train_end
            else "validation"
            if fraction < validation_end
            else "test"
        )
        return assigned == self.split

    def _shuffle_records(
        self, records: Iterator[Mapping[str, Any]]
    ) -> Iterator[Mapping[str, Any]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        buffer: list[Mapping[str, Any]] = []
        source = iter(records)

        for _ in range(self.shuffle_buffer_size):
            try:
                buffer.append(next(source))
            except StopIteration:
                break

        while buffer:
            index = rng.randrange(len(buffer))
            selected = buffer.pop(index)
            yield selected
            try:
                buffer.append(next(source))
            except StopIteration:
                pass

    def _load_sample(self, record: Mapping[str, Any]) -> StreamingSample:
        try:
            loaded = (
                self.loader(record)
                if self.loader is not None
                else _default_loader(record, self._manifest_directory)
            )
        except ManifestRecordError:
            raise
        except Exception as exc:
            raise ManifestRecordError(
                f"sample {record['sample_id']!r} could not be loaded: {exc}"
            ) from exc
        if not isinstance(loaded, Mapping):
            raise ManifestRecordError(
                f"sample {record['sample_id']!r} loader must return a mapping"
            )

        arrays_value = loaded.get("arrays")
        if isinstance(arrays_value, Mapping):
            arrays = dict(arrays_value)
        else:
            arrays = {
                key: value
                for key, value in loaded.items()
                if key not in {"sample_id", "metadata", "split", "sample"}
            }

        metadata_value = loaded.get("metadata", record.get("metadata", {}))
        if metadata_value is None:
            metadata: Mapping[str, Any] = {}
        elif isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            raise ManifestRecordError(
                f"sample {record['sample_id']!r} metadata must be a mapping"
            )

        return StreamingSample(
            sample_id=str(record["sample_id"]),
            arrays=arrays,
            uncertainty=_field(loaded, arrays, record, "uncertainty"),
            validity_mask=_field(loaded, arrays, record, "validity_mask"),
            interpolation_mask=_field(loaded, arrays, record, "interpolation_mask"),
            timestamps=_field(loaded, arrays, record, "timestamps", "times"),
            wavelengths=_field(loaded, arrays, record, "wavelengths", "wavelength"),
            metadata=metadata,
            manifest_record=record,
        )


def _validate_record(
    record: Any, line_number: int | None
) -> Mapping[str, Any]:
    location = f"line {line_number}" if line_number is not None else "record"
    if not isinstance(record, Mapping):
        raise ManifestRecordError(f"manifest {location} must be a JSON object")
    sample_id = record.get("sample_id")
    if sample_id is None or not str(sample_id).strip():
        raise ManifestRecordError(f"manifest {location} must contain a non-empty sample_id")
    return record


def _default_loader(
    record: Mapping[str, Any], manifest_directory: Path
) -> Mapping[str, Any]:
    arrays_value = record.get("arrays", record.get("array_paths"))
    if not isinstance(arrays_value, Mapping):
        return record

    arrays: dict[str, Any] = {}
    for name, value in arrays_value.items():
        arrays[str(name)] = _load_array_value(value, manifest_directory)
    loaded = dict(record)
    loaded["arrays"] = arrays
    return loaded


def _load_array_value(value: Any, manifest_directory: Path) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    suffix = path.suffix.lower()
    if suffix not in {".npy", ".npz"}:
        raise ManifestRecordError(
            f"array path {path} must use .npy or .npz, or be handled by a custom loader"
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise ManifestRecordError(
            f"NumPy is required to load array path {path}; provide a custom loader"
        ) from exc

    try:
        if suffix == ".npy":
            return np.load(path, mmap_mode="r", allow_pickle=False)
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise ManifestRecordError(f"array path {path} could not be loaded: {exc}") from exc


def _field(
    loaded: Mapping[str, Any],
    arrays: Mapping[str, Any],
    record: Mapping[str, Any],
    *names: str,
) -> Any:
    for source in (loaded, arrays, record):
        value = _first_present(source, *names)
        if value is not None:
            return value
    return None


def _first_present(source: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
    return None
