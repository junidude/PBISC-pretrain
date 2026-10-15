"""Faster exact row gathering from the E2 CSR Zarr shards.

The Model 7 ``_csr_rows`` reads ``indptr`` once per contiguous row run and
decodes every touched ``data``/``indices`` chunk sequentially through Zarr.
For the all-cell recipe streams the requested rows are scattered, so a K-row
block issues ~K ``indptr`` reads (each re-decoding the same chunk) and then
decodes most chunks of the shard one after another.  Both dominate block
loading (measured 2026-09-16: ~5 s per block, GPU mostly idle).

This module produces the *identical* CSR matrix while

* reading the ``indptr`` span ``[first_row, last_row + 2)`` exactly once, and
* decoding the touched chunks in a thread pool straight from the chunk files
  when the array uses the plain ``bytes`` + ``zstd`` Zarr v3 codec chain
  (numcodecs' zstd decoder releases the GIL, so threads scale).

Any array whose layout is not recognised (Zarr v2, other codecs, sharding,
missing chunk file, unexpected decoded length) transparently falls back to
ordinary Zarr slice reads, so correctness never depends on the fast path.
Enable in the trainer with ``PBISC_SDD_FAST_ROWS=<threads>``.
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ENV_VAR = "PBISC_SDD_FAST_ROWS"
MAX_THREADS = 32


def fast_rows_threads() -> int:
    """Return the configured thread count, or 0 when the fast path is off."""

    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_VAR} must be an integer thread count, got {raw!r}") from error
    if value < 0 or value > MAX_THREADS:
        raise ValueError(f"{ENV_VAR} must be in [0, {MAX_THREADS}], got {value}")
    return value


def _zstd_decoder():
    try:
        import zstandard

        def decode(payload: bytes, max_output: int) -> bytes:
            return zstandard.ZstdDecompressor().decompress(payload, max_output_size=max_output)

        return decode
    except ImportError:  # pragma: no cover - depends on the environment
        from numcodecs import Zstd

        def decode(payload: bytes, max_output: int) -> bytes:  # noqa: ARG001
            return bytes(Zstd().decode(payload))

        return decode


class ChunkSource:
    """Decode whole chunks of a 1-D Zarr array, bypassing Zarr when it is safe."""

    def __init__(self, array: Any) -> None:
        self.array = array
        self.chunk_size = int(getattr(array, "chunks", (0,))[0] or 0)
        self.length = int(array.shape[0])
        self.raw_dir: Path | None = None
        self.raw_dtype: np.dtype[Any] | None = None
        self._decode = None
        self._probe_raw_layout()

    def _probe_raw_layout(self) -> None:
        metadata = getattr(self.array, "metadata", None)
        store = getattr(self.array, "store", None)
        root = getattr(store, "root", None)
        if metadata is None or root is None or self.chunk_size <= 0:
            return
        if getattr(metadata, "zarr_format", None) != 3:
            return
        codecs = list(getattr(metadata, "codecs", ()) or ())
        if len(codecs) != 2:
            return
        bytes_codec, zstd_codec = codecs
        if type(bytes_codec).__name__ != "BytesCodec" or type(zstd_codec).__name__ != "ZstdCodec":
            return
        endian = getattr(bytes_codec, "endian", None)
        endian_name = getattr(endian, "value", endian)
        if endian_name not in (None, "little"):
            return
        encoding = getattr(metadata, "chunk_key_encoding", None)
        if type(encoding).__name__ != "DefaultChunkKeyEncoding" or getattr(encoding, "separator", "/") != "/":
            return
        base = np.dtype(str(self.array.dtype))
        if base.kind not in "iuf" or base.itemsize == 0:
            return
        self.raw_dir = Path(str(root)) / str(self.array.path) / "c"
        self.raw_dtype = base.newbyteorder("<") if base.itemsize > 1 else base
        self._decode = _zstd_decoder()

    @property
    def raw_capable(self) -> bool:
        return self.raw_dir is not None

    def _zarr_chunk(self, chunk_id: int) -> np.ndarray:
        start = chunk_id * self.chunk_size
        stop = min(start + self.chunk_size, self.length)
        return np.asarray(self.array[start:stop])

    def chunk(self, chunk_id: int) -> np.ndarray:
        """Return the decoded values of one chunk (edge chunk trimmed)."""

        start = chunk_id * self.chunk_size
        stop = min(start + self.chunk_size, self.length)
        if not self.raw_capable:
            return self._zarr_chunk(chunk_id)
        assert self.raw_dir is not None and self.raw_dtype is not None and self._decode is not None
        path = self.raw_dir / str(chunk_id)
        try:
            payload = path.read_bytes()
        except OSError:
            return self._zarr_chunk(chunk_id)
        decoded = self._decode(payload, self.chunk_size * self.raw_dtype.itemsize)
        values = np.frombuffer(decoded, dtype=self.raw_dtype)
        if values.shape[0] < stop - start:
            return self._zarr_chunk(chunk_id)
        return values[: stop - start]


def csr_rows_fast(
    group: Any,
    rows: np.ndarray,
    n_cols: int,
    *,
    threads: int = 4,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> sparse.csr_matrix:
    """Gather ``rows`` of a CSR Zarr group exactly like Model 7 ``_csr_rows``."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("CSR row indices must be one-dimensional")
    if rows.size == 0:
        return sparse.csr_matrix((0, int(n_cols)), dtype=dtype)
    if np.any(rows < 0):
        raise ValueError("CSR row indices must be non-negative")
    threads = max(1, int(threads))
    sort_order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[sort_order]
    indptr_array = group["indptr"]
    first = int(sorted_rows[0])
    last = int(sorted_rows[-1])
    if last + 1 >= int(indptr_array.shape[0]):
        raise IndexError("CSR row index exceeds the shard")
    # One read for the whole span: at most (last - first + 2) int64 values.
    span = np.asarray(indptr_array[first : last + 2], dtype=np.int64)
    offsets = sorted_rows - first
    starts = span[offsets]
    ends = span[offsets + 1]
    lengths = ends - starts
    out_indptr = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
    nnz = int(out_indptr[-1])
    data = np.empty(nnz, dtype=dtype)
    indices = np.empty(nnz, dtype=np.int32)

    data_source = ChunkSource(group["data"])
    index_source = ChunkSource(group["indices"])
    chunk_size = data_source.chunk_size
    if chunk_size <= 0 or index_source.chunk_size != chunk_size:
        cursor = 0
        data_array = group["data"]
        indices_array = group["indices"]
        for start, end in zip(starts, ends, strict=True):
            length = int(end - start)
            if length:
                data[cursor : cursor + length] = np.asarray(data_array[int(start) : int(end)], dtype=dtype)
                indices[cursor : cursor + length] = np.asarray(indices_array[int(start) : int(end)], dtype=np.int32)
                cursor += length
    else:
        chunk_segments: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for row_number, (start, end) in enumerate(zip(starts, ends, strict=True)):
            cursor = int(start)
            while cursor < int(end):
                chunk_id = cursor // chunk_size
                chunk_end = min(int(end), (chunk_id + 1) * chunk_size)
                chunk_segments[chunk_id].append((row_number, cursor, chunk_end))
                cursor = chunk_end
        write_offsets = out_indptr[:-1].copy()
        chunk_ids = list(chunk_segments)

        def fetch(chunk_id: int) -> tuple[int, np.ndarray, np.ndarray]:
            return chunk_id, data_source.chunk(chunk_id), index_source.chunk(chunk_id)

        if threads > 1 and len(chunk_ids) > 1:
            pool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=min(threads, len(chunk_ids)))
            fetched = pool.map(fetch, chunk_ids)
        else:
            pool = None
            fetched = map(fetch, chunk_ids)
        try:
            for chunk_id, data_chunk, index_chunk in fetched:
                chunk_start = chunk_id * chunk_size
                for row_number, start, end in chunk_segments[chunk_id]:
                    length = int(end) - int(start)
                    output_start = int(write_offsets[row_number])
                    output_end = output_start + length
                    relative_start = int(start) - chunk_start
                    relative_end = relative_start + length
                    data[output_start:output_end] = data_chunk[relative_start:relative_end]
                    indices[output_start:output_end] = index_chunk[relative_start:relative_end]
                    write_offsets[row_number] = output_end
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
    result = sparse.csr_matrix((data, indices, out_indptr), shape=(len(rows), int(n_cols)), dtype=dtype)
    if not np.array_equal(sort_order, np.arange(len(rows), dtype=np.int64)):
        result = result[np.argsort(sort_order, kind="mergesort")].tocsr()
    return result


def load_sparse_refs_fast(
    base: Any,
    refs: list[tuple[int, int]],
    pools: list[dict[str, Any]],
    *,
    threads: int,
) -> sparse.csr_matrix:
    """Drop-in for Model 7 ``E2FullDataModule._load_sparse_refs`` using ``csr_rows_fast``."""

    by_segment: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for position, (segment, row) in enumerate(refs):
        by_segment[int(segment)].append((int(position), int(row)))
    matrices: list[sparse.csr_matrix] = []
    original_positions: list[int] = []
    n_cols = int(base.source_n_genes)
    for segment, positioned_rows in by_segment.items():
        ordered = sorted(positioned_rows, key=lambda item: item[1])
        rows = np.asarray([row for _, row in ordered], dtype=np.int64)
        group = base._zarr_group(str(pools[segment]["count_zarr"]))
        matrices.append(csr_rows_fast(group, rows, n_cols, threads=threads))
        original_positions.extend(position for position, _ in ordered)
    if not matrices:
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)
    combined = matrices[0] if len(matrices) == 1 else sparse.vstack(matrices, format="csr")
    restore_order = np.argsort(np.asarray(original_positions, dtype=np.int64), kind="mergesort")
    return combined[restore_order].tocsr()


__all__ = ["ENV_VAR", "ChunkSource", "csr_rows_fast", "fast_rows_threads", "load_sparse_refs_fast"]
