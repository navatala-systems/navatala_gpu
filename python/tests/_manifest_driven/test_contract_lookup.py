# SPDX-License-Identifier: Apache-2.0
"""Verify the internal _meta module exposes manifest_id() and contract_for().

These tests live under the regen-managed test tree and are auto-emitted from
the binding manifest. Update by regenerating, not by editing in place.
"""
PACKAGE_NAME = "navatala_gpu"

REPRESENTATIVE_CONTRACTS = {
    "linalg.axpy": "navatala://python/linalg/axpy",
    "dataframe.reduce_sum": "navatala://python/dataframe/reduce_sum",
    "ml.mean_squared_error": "navatala://python/ml/mean_squared_error",
    "sparse.csr_spmv": "navatala://python/sparse/csr_spmv",
    "graph.out_degree": "navatala://python/graph/out_degree",
}

def test_meta_module_is_not_in_public_all():
    pkg = __import__(PACKAGE_NAME)
    public_all = list(getattr(pkg, "__all__", []))
    assert "_meta" not in public_all, "internal `_meta` must not be in __all__"

def test_meta_module_is_importable_via_attribute():
    pkg = __import__(PACKAGE_NAME)
    meta = getattr(pkg, "_meta", None)
    assert meta is not None, "navatala_gpu._meta must be reachable from the package"

def test_manifest_id_resolves_to_non_empty_string():
    pkg = __import__(PACKAGE_NAME)
    mid = pkg._meta.manifest_id()
    assert isinstance(mid, str) and mid, f"manifest_id() returned {mid!r}"

def test_contract_for_resolves_representative_bindings():
    pkg = __import__(PACKAGE_NAME)
    for symbol, expected in REPRESENTATIVE_CONTRACTS.items():
        got = pkg._meta.contract_for(symbol)
        assert got == expected, f"{symbol!r} resolved to {got!r}, expected {expected!r}"

def test_contract_for_unknown_symbol_returns_none():
    pkg = __import__(PACKAGE_NAME)
    assert pkg._meta.contract_for("not.a.real.binding") is None
