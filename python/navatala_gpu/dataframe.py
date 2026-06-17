# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import supports as _runtime_supports, validate_call

__navatala_module__ = "dataframe"
_BINDINGS = ("reduce_sum", "reduce_sum_i32_to_i64", "reduce_min_i32", "reduce_max_i32", "fill_null", "fill_null_i32", "gather", "gather_i32", "scatter", "scatter_i32", "compact_by_mask", "compact_by_mask_i32", "coalesce", "copy_if_valid", "inclusive_scan_sum_valid_prefix", "inclusive_scan_sum_i32_to_i64_skip_nulls", "count_valid", "is_valid", "is_null", "fill_constant", "fill_constant_i32", "fill_range", "fill_validity_all_valid", "fill_validity_all_null")
__all__ = ("reduce_sum", "reduce_sum_i32_to_i64", "reduce_min_i32", "reduce_max_i32", "fill_null", "fill_null_i32", "gather", "gather_i32", "scatter", "scatter_i32", "compact_by_mask", "compact_by_mask_i32", "coalesce", "copy_if_valid", "inclusive_scan_sum_valid_prefix", "inclusive_scan_sum_i32_to_i64_skip_nulls", "count_valid", "is_valid", "is_null", "fill_constant", "fill_constant_i32", "fill_range", "fill_validity_all_valid", "fill_validity_all_null", "list_bindings", "supports")

def list_bindings() -> tuple[str, ...]:
    """Return the public operation names exposed by this module."""
    return _BINDINGS

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    """Return whether this module exposes an operation for the requested backend/dtype."""
    if operation not in _BINDINGS:
        return False
    return _runtime_supports(f"{__navatala_module__}.{operation}", backend=backend, dtype=dtype)

def reduce_sum(data, input_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compute a null-aware float32 column sum.

    The input_valid bitmap uses uint32 words with one bit per row. output is a scalar float32 tensor; output_valid bit 0 is set when at least one element is valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "reduce_sum", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "reduce_sum", data, input_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def reduce_sum_i32_to_i64(data, input_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compute a null-aware int32 column sum widened to int64.

    The input_valid bitmap uses uint32 words with one bit per row. output is a scalar int64 tensor; output_valid bit 0 is set when at least one element is valid. Nulls are skipped and values are widened before accumulation.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, int64, uint32
    - hip: int32, int64, uint32
    - opencl: int32, int64, uint32
    - vulkan: int32, int64, uint32
    - metal: int32, int64, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "reduce_sum_i32_to_i64", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "reduce_sum_i32_to_i64", data, input_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def reduce_min_i32(data, input_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compute a null-aware int32 column minimum.

    The input_valid bitmap uses uint32 words with one bit per row. output is a scalar int32 tensor; output_valid bit 0 is set when at least one element is valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, uint32
    - hip: int32, uint32
    - opencl: int32, uint32
    - vulkan: int32, uint32
    - metal: int32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "reduce_min_i32", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "reduce_min_i32", data, input_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def reduce_max_i32(data, input_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compute a null-aware int32 column maximum.

    The input_valid bitmap uses uint32 words with one bit per row. output is a scalar int32 tensor; output_valid bit 0 is set when at least one element is valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, uint32
    - hip: int32, uint32
    - opencl: int32, uint32
    - vulkan: int32, uint32
    - metal: int32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "reduce_max_i32", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "reduce_max_i32", data, input_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def fill_null(data, input_valid, output, output_valid, fill_value, count, backend="auto", *, queue=None, sync=False):
    """
    Replace null float32 column entries with a scalar fill value.

    The input_valid bitmap uses uint32 words with one bit per row. output_valid is overwritten so every row in [0, count) is valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_null", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"fill_value": fill_value, "count": count, "backend": backend})
    return call_extension("dataframe", "fill_null", data, input_valid, output, output_valid, fill_value, count, backend, queue=queue, sync=sync)

def fill_null_i32(data, input_valid, output, output_valid, fill_value, count, backend="auto", *, queue=None, sync=False):
    """
    Replace null int32 column entries with a scalar fill value.

    The input_valid bitmap uses uint32 words with one bit per row. output_valid is overwritten so every row in [0, count) is valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, uint32
    - hip: int32, uint32
    - opencl: int32, uint32
    - vulkan: int32, uint32
    - metal: int32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_null_i32", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"fill_value": fill_value, "count": count, "backend": backend})
    return call_extension("dataframe", "fill_null_i32", data, input_valid, output, output_valid, fill_value, count, backend, queue=queue, sync=sync)

def gather(data, input_valid, indices, output, output_valid, input_count, output_count, backend="auto", *, queue=None, sync=False):
    """
    Gather float32 DataFrame values by int32 row indices.

    Invalid or out-of-range source rows become null output rows. output_valid is overwritten with ceil(output_count / 32) uint32 words.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32, int32
    - hip: float32, uint32, int32
    - opencl: float32, uint32, int32
    - vulkan: float32, uint32, int32
    - metal: float32, uint32, int32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "gather", {"data": data, "input_valid": input_valid, "indices": indices, "output": output, "output_valid": output_valid}, {"input_count": input_count, "output_count": output_count, "backend": backend})
    return call_extension("dataframe", "gather", data, input_valid, indices, output, output_valid, input_count, output_count, backend, queue=queue, sync=sync)

def gather_i32(data, input_valid, indices, output, output_valid, input_count, output_count, backend="auto", *, queue=None, sync=False):
    """
    Gather int32 DataFrame values by int32 row indices.

    Out-of-range indices and null source rows produce output value 0 and clear the corresponding output validity bit.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, uint32
    - hip: int32, uint32
    - opencl: int32, uint32
    - vulkan: int32, uint32
    - metal: int32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "gather_i32", {"data": data, "input_valid": input_valid, "indices": indices, "output": output, "output_valid": output_valid}, {"input_count": input_count, "output_count": output_count, "backend": backend})
    return call_extension("dataframe", "gather_i32", data, input_valid, indices, output, output_valid, input_count, output_count, backend, queue=queue, sync=sync)

def scatter(data, input_valid, indices, output, output_valid, input_count, output_count, backend="auto", *, queue=None, sync=False):
    """
    Scatter float32 DataFrame values by int32 destination indices.

    For every in-range index, output[indices[i]] receives data[i]. Valid source rows set the destination validity bit; invalid source rows leave validity unchanged. Duplicate destination indices have contract-level undefined ordering.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32, int32
    - hip: float32, uint32, int32
    - opencl: float32, uint32, int32
    - vulkan: float32, uint32, int32
    - metal: float32, uint32, int32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "scatter", {"data": data, "input_valid": input_valid, "indices": indices, "output": output, "output_valid": output_valid}, {"input_count": input_count, "output_count": output_count, "backend": backend})
    return call_extension("dataframe", "scatter", data, input_valid, indices, output, output_valid, input_count, output_count, backend, queue=queue, sync=sync)

def scatter_i32(data, input_valid, indices, output, output_valid, input_count, output_count, backend="auto", *, queue=None, sync=False):
    """
    Scatter int32 DataFrame values by int32 destination indices.

    For every in-range index, output[indices[i]] receives data[i]. Valid source rows set the destination validity bit; invalid source rows leave validity unchanged. Duplicate destination indices have contract-level undefined ordering.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, uint32
    - hip: int32, uint32
    - opencl: int32, uint32
    - vulkan: int32, uint32
    - metal: int32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "scatter_i32", {"data": data, "input_valid": input_valid, "indices": indices, "output": output, "output_valid": output_valid}, {"input_count": input_count, "output_count": output_count, "backend": backend})
    return call_extension("dataframe", "scatter_i32", data, input_valid, indices, output, output_valid, input_count, output_count, backend, queue=queue, sync=sync)

def compact_by_mask(data, input_valid, mask, prefix_sum, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compact float32 DataFrame values by a uint32 selection mask.

    Rows with mask[i] == 1 are copied to output[prefix_sum[i]]. Source validity bits are propagated to output_valid, which is overwritten.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "compact_by_mask", {"data": data, "input_valid": input_valid, "mask": mask, "prefix_sum": prefix_sum, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "compact_by_mask", data, input_valid, mask, prefix_sum, output, output_valid, count, backend, queue=queue, sync=sync)

def compact_by_mask_i32(data, input_valid, mask, prefix_sum, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compact int32 DataFrame values by a uint32 selection mask.

    Rows with mask[i] == 1 are copied to output[prefix_sum[i]]. Source validity bits are propagated to output_valid, which is overwritten.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, uint32
    - hip: int32, uint32
    - opencl: int32, uint32
    - vulkan: int32, uint32
    - metal: int32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "compact_by_mask_i32", {"data": data, "input_valid": input_valid, "mask": mask, "prefix_sum": prefix_sum, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "compact_by_mask_i32", data, input_valid, mask, prefix_sum, output, output_valid, count, backend, queue=queue, sync=sync)

def coalesce(col1, col1_valid, col2, col2_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Select the first non-null value from two float32 DataFrame columns.

    Rows prefer col1 when col1_valid is set; otherwise they use col2 when col2_valid is set. output_valid is overwritten with ceil(count / 32) uint32 words.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "coalesce", {"col1": col1, "col1_valid": col1_valid, "col2": col2, "col2_valid": col2_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "coalesce", col1, col1_valid, col2, col2_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def copy_if_valid(src, src_valid, dst_valid, dst, out_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Copy float32 values where source and destination rows are both valid.

    dst is updated in place only at the intersection of src_valid and dst_valid; other dst rows are unchanged. out_valid is overwritten with that intersection bitmap.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "copy_if_valid", {"src": src, "src_valid": src_valid, "dst_valid": dst_valid, "dst": dst, "out_valid": out_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "copy_if_valid", src, src_valid, dst_valid, dst, out_valid, count, backend, queue=queue, sync=sync)

def inclusive_scan_sum_valid_prefix(data, input_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compute a nullable float32 inclusive prefix sum with prefix-valid null propagation for up to 256 rows.

    Null rows contribute zero to the running sum. output_valid is set only for prefixes where every input row up to that point is valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "inclusive_scan_sum_valid_prefix", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "inclusive_scan_sum_valid_prefix", data, input_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def inclusive_scan_sum_i32_to_i64_skip_nulls(data, input_valid, output, output_valid, count, backend="auto", *, queue=None, sync=False):
    """
    Compute a skip-null int32 inclusive prefix sum widened to int64 for up to 256 rows.

    Null rows contribute zero to the running sum and output_valid follows input_valid. output uses int64 values to avoid int32 cumulative-sum overflow.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32, int64, uint32
    - hip: int32, int64, uint32
    - opencl: int32, int64, uint32
    - vulkan: int32, int64, uint32
    - metal: int32, int64, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "inclusive_scan_sum_i32_to_i64_skip_nulls", {"data": data, "input_valid": input_valid, "output": output, "output_valid": output_valid}, {"count": count, "backend": backend})
    return call_extension("dataframe", "inclusive_scan_sum_i32_to_i64_skip_nulls", data, input_valid, output, output_valid, count, backend, queue=queue, sync=sync)

def count_valid(input_valid, output, count, backend="auto", *, queue=None, sync=False):
    """
    Count valid rows in a uint32 validity bitmap.

    The input_valid bitmap uses uint32 words with one bit per row. output is a scalar uint32 tensor.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32
    - hip: uint32
    - opencl: uint32
    - vulkan: uint32
    - metal: uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "count_valid", {"input_valid": input_valid, "output": output}, {"count": count, "backend": backend})
    return call_extension("dataframe", "count_valid", input_valid, output, count, backend, queue=queue, sync=sync)

def is_valid(input_valid, output, count, backend="auto", *, queue=None, sync=False):
    """
    Expand a validity bitmap into a uint8 valid mask.

    input_valid is a rank-1 uint32 bitmap with ceil(count / 32) words. output is a rank-1 uint8 tensor of length count where 1 means valid.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32, uint8
    - hip: uint32, uint8
    - opencl: uint32, uint8
    - vulkan: uint32, uint8
    - metal: uint32, uint8

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "is_valid", {"input_valid": input_valid, "output": output}, {"count": count, "backend": backend})
    return call_extension("dataframe", "is_valid", input_valid, output, count, backend, queue=queue, sync=sync)

def is_null(input_valid, output, count, backend="auto", *, queue=None, sync=False):
    """
    Expand a validity bitmap into a uint8 null mask.

    input_valid is a rank-1 uint32 bitmap with ceil(count / 32) words. output is a rank-1 uint8 tensor of length count where 1 means null.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32, uint8
    - hip: uint32, uint8
    - opencl: uint32, uint8
    - vulkan: uint32, uint8
    - metal: uint32, uint8

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "is_null", {"input_valid": input_valid, "output": output}, {"count": count, "backend": backend})
    return call_extension("dataframe", "is_null", input_valid, output, count, backend, queue=queue, sync=sync)

def fill_constant(output, value, count, backend="auto", *, queue=None, sync=False):
    """
    Fill a float32 output tensor with a scalar value.

    output is a rank-1 float32 tensor with length count. value is converted to float32 by the runtime ABI.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_constant", {"output": output}, {"value": value, "count": count, "backend": backend})
    return call_extension("dataframe", "fill_constant", output, value, count, backend, queue=queue, sync=sync)

def fill_constant_i32(output, value, count, backend="auto", *, queue=None, sync=False):
    """
    Fill an int32 output tensor with a scalar value.

    output is a rank-1 int32 tensor with length count. value must fit in the int32 range.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32
    - hip: int32
    - opencl: int32
    - vulkan: int32
    - metal: int32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_constant_i32", {"output": output}, {"value": value, "count": count, "backend": backend})
    return call_extension("dataframe", "fill_constant_i32", output, value, count, backend, queue=queue, sync=sync)

def fill_range(output, count, start=0, backend="auto", *, queue=None, sync=False):
    """
    Fill an int32 DataFrame column with a sequential range.

    output[i] = start + i for count rows. The binding rejects ranges that cannot be represented as int32 values.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: int32
    - hip: int32
    - opencl: int32
    - vulkan: int32
    - metal: int32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_range", {"output": output}, {"count": count, "start": start, "backend": backend})
    return call_extension("dataframe", "fill_range", output, count, start, backend, queue=queue, sync=sync)

def fill_validity_all_valid(output, count, backend="auto", *, queue=None, sync=False):
    """
    Fill a uint32 validity bitmap so all rows are valid.

    output is a rank-1 uint32 tensor with ceil(count / 32) words. Unused trailing bits are zero.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32
    - hip: uint32
    - opencl: uint32
    - vulkan: uint32
    - metal: uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_validity_all_valid", {"output": output}, {"count": count, "backend": backend})
    return call_extension("dataframe", "fill_validity_all_valid", output, count, backend, queue=queue, sync=sync)

def fill_validity_all_null(output, count, backend="auto", *, queue=None, sync=False):
    """
    Fill a uint32 validity bitmap so all rows are null.

    output is a rank-1 uint32 tensor with ceil(count / 32) words.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32
    - hip: uint32
    - opencl: uint32
    - vulkan: uint32
    - metal: uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("dataframe", "fill_validity_all_null", {"output": output}, {"count": count, "backend": backend})
    return call_extension("dataframe", "fill_validity_all_null", output, count, backend, queue=queue, sync=sync)