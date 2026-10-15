"""Exactness tests for the fast CSR row gatherer against plain Zarr reads."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

zarr = pytest.importorskip("zarr")
from pbisc_sdd import fast_rows  # noqa: E402


def _reference_rows(group, rows: np.ndarray, n_cols: int) -> sparse.csr_matrix:
    """Row-by-row gather through ordinary Zarr slicing (the ground truth)."""

    indptr = np.asarray(group["indptr"][:], dtype=np.int64)
    parts = []
    for row in rows:
        start, end = int(indptr[row]), int(indptr[row + 1])
        parts.append(
            sparse.csr_matrix(
                (
                    np.asarray(group["data"][start:end], dtype=np.float32),
                    np.asarray(group["indices"][start:end], dtype=np.int32),
                    np.array([0, end - start], dtype=np.int64),
                ),
                shape=(1, n_cols),
            )
        )
    if not parts:
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)
    return sparse.vstack(parts, format="csr")


def _random_csr(rng: np.random.Generator, n_rows: int, n_cols: int, density: float) -> sparse.csr_matrix:
    matrix = sparse.random(n_rows, n_cols, density=density, format="csr", random_state=rng, dtype=np.float64)
    matrix.data = rng.integers(1, 50, size=matrix.data.shape).astype(np.float64)
    matrix.sort_indices()
    return matrix


def _write_group(path: Path, matrix: sparse.csr_matrix, *, zarr_format: int, chunk: int, codec: str = "zstd"):
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    arrays = {
        "data": matrix.data.astype(np.int64),
        "indices": matrix.indices.astype(np.int32),
        "indptr": matrix.indptr.astype(np.int64),
    }
    for name, values in arrays.items():
        kwargs = {}
        if zarr_format == 3:
            if codec == "zstd":
                kwargs["compressors"] = zarr.codecs.ZstdCodec(level=0)
            else:
                kwargs["compressors"] = zarr.codecs.BloscCodec()
        else:
            from numcodecs import Blosc

            kwargs["compressor"] = Blosc(cname="lz4", clevel=5)
        array = group.create_array(name, shape=values.shape, dtype=values.dtype, chunks=(chunk,), **kwargs)
        array[:] = values
    return zarr.open_group(str(path), mode="r")


def _assert_same(a: sparse.csr_matrix, b: sparse.csr_matrix) -> None:
    a = a.tocsr(); b = b.tocsr()
    a.sort_indices(); b.sort_indices()
    assert a.shape == b.shape
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data)


ROW_CASES = [
    np.array([], dtype=np.int64),
    np.array([0]),
    np.array([39]),
    np.array([5, 5, 5]),
    np.array([7, 3, 39, 0, 12, 3]),
    np.arange(40),
    np.array([38, 1, 20, 21, 22, 2]),
]


@pytest.mark.parametrize("zarr_format,codec,expect_raw", [(3, "zstd", True), (3, "blosc", False), (2, "blosc", False)])
def test_fast_rows_match_reference_and_use_raw_only_for_plain_zstd_v3(tmp_path: Path, zarr_format, codec, expect_raw) -> None:
    rng = np.random.default_rng(20260916)
    matrix = _random_csr(rng, 40, 25, density=0.3)
    group = _write_group(tmp_path / f"shard_v{zarr_format}_{codec}.zarr", matrix, zarr_format=zarr_format, chunk=7, codec=codec)
    source = fast_rows.ChunkSource(group["data"])
    assert source.raw_capable is expect_raw
    assert source.chunk_size == 7
    for rows in ROW_CASES:
        for threads in (1, 3):
            got = fast_rows.csr_rows_fast(group, rows, 25, threads=threads)
            _assert_same(got, _reference_rows(group, rows, 25))


def test_fast_rows_edge_chunk_and_wide_span(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    matrix = _random_csr(rng, 300, 60, density=0.15)  # nnz not a multiple of the chunk
    group = _write_group(tmp_path / "wide.zarr", matrix, zarr_format=3, chunk=101)
    assert fast_rows.ChunkSource(group["data"]).raw_capable
    rows = rng.choice(300, size=120, replace=False)
    _assert_same(fast_rows.csr_rows_fast(group, rows, 60, threads=4), _reference_rows(group, rows, 60))
    rows = np.array([299, 0, 150])
    _assert_same(fast_rows.csr_rows_fast(group, rows, 60, threads=4), _reference_rows(group, rows, 60))


def test_missing_chunk_file_falls_back_to_zarr(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    matrix = _random_csr(rng, 30, 20, density=0.4)
    group = _write_group(tmp_path / "fb.zarr", matrix, zarr_format=3, chunk=9)
    source = fast_rows.ChunkSource(group["data"])
    assert source.raw_capable
    source.raw_dir = tmp_path / "nowhere"  # simulate an unreadable chunk directory
    expected = np.asarray(group["data"][9:18])
    assert np.array_equal(source.chunk(1), expected)


def test_out_of_range_row_is_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    group = _write_group(tmp_path / "oob.zarr", _random_csr(rng, 10, 8, density=0.5), zarr_format=3, chunk=5)
    with pytest.raises(IndexError):
        fast_rows.csr_rows_fast(group, np.array([10]), 8)
    with pytest.raises(ValueError):
        fast_rows.csr_rows_fast(group, np.array([-1]), 8)


def test_env_threads_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fast_rows.ENV_VAR, raising=False)
    assert fast_rows.fast_rows_threads() == 0
    monkeypatch.setenv(fast_rows.ENV_VAR, "4")
    assert fast_rows.fast_rows_threads() == 4
    monkeypatch.setenv(fast_rows.ENV_VAR, "0")
    assert fast_rows.fast_rows_threads() == 0
    for bad in ("x", "-1", "99"):
        monkeypatch.setenv(fast_rows.ENV_VAR, bad)
        with pytest.raises(ValueError):
            fast_rows.fast_rows_threads()


class _FakeBase:
    """Minimal stand-in for Model 7 E2FullDataModule for load_sparse_refs_fast."""

    def __init__(self, groups: dict[str, object], n_cols: int) -> None:
        self._groups = groups
        self.source_n_genes = n_cols

    def _zarr_group(self, relative_path: str):
        return self._groups[relative_path]


def test_load_sparse_refs_fast_restores_ref_order_across_segments(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    m0 = _random_csr(rng, 20, 15, density=0.3)
    m1 = _random_csr(rng, 25, 15, density=0.3)
    g0 = _write_group(tmp_path / "s0.zarr", m0, zarr_format=3, chunk=6)
    g1 = _write_group(tmp_path / "s1.zarr", m1, zarr_format=2, chunk=6)
    base = _FakeBase({"a": g0, "b": g1}, 15)
    pools = [{"count_zarr": "a"}, {"count_zarr": "b"}]
    refs = [(1, 24), (0, 3), (1, 0), (0, 19), (0, 3), (1, 7)]
    got = fast_rows.load_sparse_refs_fast(base, refs, pools, threads=2)
    expected = sparse.vstack([(m1 if s == 1 else m0)[r] for s, r in refs], format="csr").astype(np.float32)
    _assert_same(got, expected)
    assert fast_rows.load_sparse_refs_fast(base, [], pools, threads=2).shape == (0, 15)
