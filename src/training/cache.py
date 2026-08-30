"""Small, disk-backed cache for rendered synthetic training samples."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import numpy as np


MAX_CACHE_ENTRIES = 64
DEFAULT_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ENTRY_BYTES = 64 * 1024 * 1024
STORAGE_CAP_BYTES = 5 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class DiskCachePayload:
    """One entry read from disk; the cache object never retains this payload."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _IndexEntry:
    filename: str
    byte_count: int


class ProceduralSyntheticCache:
    """Persist compressed synthetic samples with a tiny JSON-backed LRU index.

    The cache retains only entry names and byte counts in memory.  Full arrays
    exist only while a caller is rendering, writing, or loading a single NPZ
    payload, so configuring the maximum at 64 cannot retain 64 full scenes in
    process RAM.  All payload bytes live on the caller-provided cache
    directory (an SSD path in normal use), not in memory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_cache_entries: int = MAX_CACHE_ENTRIES,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        max_entry_bytes: int | None = None,
    ) -> None:
        if not isinstance(max_cache_entries, int) or not 1 <= max_cache_entries <= MAX_CACHE_ENTRIES:
            raise ValueError(f"max_cache_entries must be an integer in [1, {MAX_CACHE_ENTRIES}]")
        if not isinstance(max_cache_bytes, int) or not 1 <= max_cache_bytes <= STORAGE_CAP_BYTES:
            raise ValueError("max_cache_bytes must be positive and within the 5 GiB storage cap")
        if max_entry_bytes is None:
            max_entry_bytes = min(DEFAULT_MAX_ENTRY_BYTES, max_cache_bytes)
        if not isinstance(max_entry_bytes, int) or not 1 <= max_entry_bytes <= max_cache_bytes:
            raise ValueError("max_entry_bytes must be positive and no larger than max_cache_bytes")

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.json"
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._max_entry_bytes = max_entry_bytes
        self._entries = self._read_index()

    @property
    def cache_size(self) -> int:
        return len(self._entries)

    @property
    def cache_bytes(self) -> int:
        return sum(entry.byte_count for entry in self._entries.values())

    @property
    def max_cache_entries(self) -> int:
        return self._max_cache_entries

    @property
    def max_cache_bytes(self) -> int:
        return self._max_cache_bytes

    @property
    def max_entry_bytes(self) -> int:
        return self._max_entry_bytes

    def load(self, key: str) -> DiskCachePayload | None:
        """Load one compressed payload and close its archive before returning."""

        entry = self._entries.get(key)
        if entry is None:
            return None
        path = self._cache_dir / entry.filename
        try:
            with np.load(path, allow_pickle=False, mmap_mode="r") as archive:
                metadata = json.loads(str(archive["__metadata__"].item()))
                arrays = {
                    name: archive[name]
                    for name in archive.files
                    if name != "__metadata__"
                }
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            self._remove_entry(key)
            return None
        if not isinstance(metadata, dict):
            self._remove_entry(key)
            return None
        self._entries.move_to_end(key)
        self._write_index()
        return DiskCachePayload(arrays=arrays, metadata=metadata)

    def store(
        self,
        key: str,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, object],
    ) -> bool:
        """Atomically write one entry and evict oldest entries to keep bounds."""

        if not key:
            raise ValueError("cache key must be non-empty")
        if not arrays:
            raise ValueError("cache entries must contain at least one array")
        metadata_text = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
        filename = f"entry-{_key_digest(key)}.npz"
        destination = self._cache_dir / filename
        temporary = self._cache_dir / f".{filename}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    **{name: np.asarray(value) for name, value in arrays.items()},
                    __metadata__=np.asarray(metadata_text),
                )
                handle.flush()
                os.fsync(handle.fileno())
            byte_count = temporary.stat().st_size
            if byte_count > self._max_entry_bytes or byte_count > self._max_cache_bytes:
                return False
            self._remove_entry(key, write_index=False)
            self._evict_to_fit(byte_count)
            os.replace(temporary, destination)
            self._entries[key] = _IndexEntry(filename=filename, byte_count=byte_count)
            self._write_index()
            return True
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_index(self) -> OrderedDict[str, _IndexEntry]:
        if not self._index_path.is_file():
            return OrderedDict()
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
            raw_entries = payload["entries"]
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return OrderedDict()
        entries: OrderedDict[str, _IndexEntry] = OrderedDict()
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            key = raw.get("key")
            filename = raw.get("filename")
            byte_count = raw.get("byte_count")
            if (
                not isinstance(key, str)
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(byte_count, int)
                or byte_count < 1
            ):
                continue
            path = self._cache_dir / filename
            if path.is_file() and path.stat().st_size == byte_count:
                entries[key] = _IndexEntry(filename=filename, byte_count=byte_count)
        while len(entries) > self._max_cache_entries or self._entry_bytes(entries) > self._max_cache_bytes:
            entries.popitem(last=False)
        return entries

    def _evict_to_fit(self, incoming_bytes: int) -> None:
        while self._entries and (
            len(self._entries) >= self._max_cache_entries
            or self.cache_bytes + incoming_bytes > self._max_cache_bytes
        ):
            oldest_key = next(iter(self._entries))
            self._remove_entry(oldest_key, write_index=False)

    def _remove_entry(self, key: str, *, write_index: bool = True) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            path = self._cache_dir / entry.filename
            if path.is_file():
                path.unlink()
        if write_index:
            self._write_index()

    def _write_index(self) -> None:
        payload = {
            "version": 1,
            "entries": [
                {"key": key, "filename": entry.filename, "byte_count": entry.byte_count}
                for key, entry in self._entries.items()
            ],
        }
        temporary = self._cache_dir / f".index.{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._index_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _entry_bytes(entries: Mapping[str, _IndexEntry]) -> int:
        return sum(entry.byte_count for entry in entries.values())


def _key_digest(key: str) -> str:
    import hashlib

    return hashlib.sha256(key.encode("utf-8")).hexdigest()
