# SPDX-License-Identifier: Apache-2.0
"""Internal metadata for navatala_gpu.

Carries the manifest id (release/regeneration identifier) and the mapping
from public symbol names to public contract ids. NOT part of the
public API — kept underscore-prefixed so it does not appear in
``navatala_gpu.__all__``. Support tooling and bug reporters may call
``navatala_gpu._meta.manifest_id()`` and
``navatala_gpu._meta.contract_for(symbol)``.
"""
from __future__ import annotations

from typing import Optional

_MANIFEST_ID = "pyabi6-2675513039"

_CONTRACTS = {
    "linalg.axpy": "navatala://python/linalg/axpy",
    "linalg.axpy_dispatch": "navatala://python/linalg/axpy_dispatch",
    "linalg.gemm": "navatala://python/linalg/gemm",
    "linalg.nrm2": "navatala://python/linalg/nrm2",
    "dataframe.reduce_sum": "navatala://python/dataframe/reduce_sum",
    "dataframe.reduce_sum_i32_to_i64": "navatala://python/dataframe/reduce_sum_i32_to_i64",
    "dataframe.reduce_min_i32": "navatala://python/dataframe/reduce_min_i32",
    "dataframe.reduce_max_i32": "navatala://python/dataframe/reduce_max_i32",
    "dataframe.fill_null": "navatala://python/dataframe/fill_null",
    "dataframe.fill_null_i32": "navatala://python/dataframe/fill_null_i32",
    "dataframe.gather": "navatala://python/dataframe/gather",
    "dataframe.gather_i32": "navatala://python/dataframe/gather_i32",
    "dataframe.scatter": "navatala://python/dataframe/scatter",
    "dataframe.scatter_i32": "navatala://python/dataframe/scatter_i32",
    "dataframe.compact_by_mask": "navatala://python/dataframe/compact_by_mask",
    "dataframe.compact_by_mask_i32": "navatala://python/dataframe/compact_by_mask_i32",
    "dataframe.coalesce": "navatala://python/dataframe/coalesce",
    "dataframe.copy_if_valid": "navatala://python/dataframe/copy_if_valid",
    "dataframe.inclusive_scan_sum_valid_prefix": "navatala://python/dataframe/inclusive_scan_sum_valid_prefix",
    "dataframe.inclusive_scan_sum_i32_to_i64_skip_nulls": "navatala://python/dataframe/inclusive_scan_sum_i32_to_i64_skip_nulls",
    "dataframe.count_valid": "navatala://python/dataframe/count_valid",
    "dataframe.is_valid": "navatala://python/dataframe/is_valid",
    "dataframe.is_null": "navatala://python/dataframe/is_null",
    "dataframe.fill_constant": "navatala://python/dataframe/fill_constant",
    "dataframe.fill_constant_i32": "navatala://python/dataframe/fill_constant_i32",
    "dataframe.fill_range": "navatala://python/dataframe/fill_range",
    "dataframe.fill_validity_all_valid": "navatala://python/dataframe/fill_validity_all_valid",
    "dataframe.fill_validity_all_null": "navatala://python/dataframe/fill_validity_all_null",
    "ml.mean_squared_error": "navatala://python/ml/mean_squared_error",
    "ml.root_mean_squared_error": "navatala://python/ml/root_mean_squared_error",
    "ml.mean_absolute_error": "navatala://python/ml/mean_absolute_error",
    "ml.mean_absolute_percentage_error": "navatala://python/ml/mean_absolute_percentage_error",
    "ml.r2_score": "navatala://python/ml/r2_score",
    "ml.explained_variance_score": "navatala://python/ml/explained_variance_score",
    "sparse.csr_spmv": "navatala://python/sparse/csr_spmv",
    "graph.out_degree": "navatala://python/graph/out_degree",
    "graph.in_degree": "navatala://python/graph/in_degree",
}

def manifest_id() -> str:
    """Return the regeneration identifier of this build."""
    return _MANIFEST_ID

def contract_for(symbol: str) -> Optional[str]:
    """Return the public contract id for ``symbol`` (e.g. ``"linalg.axpy"``).

    Returns ``None`` if the symbol is not a public binding.
    """
    return _CONTRACTS.get(symbol)
