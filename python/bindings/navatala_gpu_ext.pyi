# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from navatala_gpu.runtime import Queue, Tensor

def axpy(x: Tensor, y: Tensor, alpha: float = 1.000000, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute y = alpha * x + y on the selected GPU backend.

    The Python binding accepts DLPack-compatible tensors and returns the updated y tensor.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    The binding returns after enqueue unless the caller requests synchronization.
    """
    ...

def axpy_dispatch(x: Tensor, y: Tensor, alpha: float = 1.000000, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute y = alpha * x + y through a dtype-dispatched binding group.

    The Python binding validates dispatch-group dtype routing before invoking the concrete AXPY implementation.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    The binding returns after enqueue unless the caller requests synchronization.
    """
    ...

def gemm(a: Tensor, b: Tensor, c: Tensor, alpha: float = 1.000000, beta: float = 0.000000, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute C = alpha * A @ B + beta * C on the selected GPU backend.

    The Python binding validates rank-2 tensors and backend dtype support before dispatch.

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
    ...

def nrm2(x: Tensor, result: Tensor, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute result[0] = ||x||_2 on the selected GPU backend.

    The result argument is an explicit scalar float32 Tensor, keeping output allocation visible to the caller.

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
    ...

def reduce_sum(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def reduce_sum_i32_to_i64(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def reduce_min_i32(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def reduce_max_i32(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_null(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, fill_value: float, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_null_i32(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, fill_value: int, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def gather(data: Tensor, input_valid: Tensor, indices: Tensor, output: Tensor, output_valid: Tensor, input_count: int, output_count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def gather_i32(data: Tensor, input_valid: Tensor, indices: Tensor, output: Tensor, output_valid: Tensor, input_count: int, output_count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def scatter(data: Tensor, input_valid: Tensor, indices: Tensor, output: Tensor, output_valid: Tensor, input_count: int, output_count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def scatter_i32(data: Tensor, input_valid: Tensor, indices: Tensor, output: Tensor, output_valid: Tensor, input_count: int, output_count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def compact_by_mask(data: Tensor, input_valid: Tensor, mask: Tensor, prefix_sum: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def compact_by_mask_i32(data: Tensor, input_valid: Tensor, mask: Tensor, prefix_sum: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def coalesce(col1: Tensor, col1_valid: Tensor, col2: Tensor, col2_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def copy_if_valid(src: Tensor, src_valid: Tensor, dst_valid: Tensor, dst: Tensor, out_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def inclusive_scan_sum_valid_prefix(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def inclusive_scan_sum_i32_to_i64_skip_nulls(data: Tensor, input_valid: Tensor, output: Tensor, output_valid: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def count_valid(input_valid: Tensor, output: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def is_valid(input_valid: Tensor, output: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def is_null(input_valid: Tensor, output: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_constant(output: Tensor, value: float, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_constant_i32(output: Tensor, value: int, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_range(output: Tensor, count: int, start: int = 0, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_validity_all_valid(output: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def fill_validity_all_null(output: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
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
    ...

def mean_squared_error(y_true: Tensor, y_pred: Tensor, result: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute float32 mean squared error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.

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
    ...

def root_mean_squared_error(y_true: Tensor, y_pred: Tensor, result: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute float32 root mean squared error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.

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
    ...

def mean_absolute_error(y_true: Tensor, y_pred: Tensor, result: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute float32 mean absolute error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.

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
    ...

def mean_absolute_percentage_error(y_true: Tensor, y_pred: Tensor, result: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute float32 mean absolute percentage error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor. y_true == 0 follows IEEE-754 division semantics.

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
    ...

def r2_score(y_true: Tensor, y_pred: Tensor, y_mean: Tensor, result: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute float32 R2 score.

    y_true and y_pred are rank-1 float32 tensors with length count. y_mean is an explicit scalar float32 tensor containing the precomputed mean of y_true; result is an explicit scalar float32 tensor. Constant y_true inputs follow IEEE-754 division semantics.

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
    ...

def explained_variance_score(y_true: Tensor, y_pred: Tensor, mean_residuals: Tensor, mean_true: Tensor, result: Tensor, count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute float32 explained variance score.

    y_true and y_pred are rank-1 float32 tensors with length count. mean_residuals and mean_true are explicit scalar float32 tensors; result is an explicit scalar float32 tensor. Constant y_true inputs follow IEEE-754 division semantics.

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
    ...

def csr_spmv(row_offsets: Tensor, col_indices: Tensor, values: Tensor, x: Tensor, y: Tensor, rows: int, cols: int, nnz: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute y = A * x for a float32 CSR sparse matrix.

    The CSR descriptor uses uint32 row_offsets/col_indices and float32 values. rows, cols, and nnz make the descriptor contract explicit at the Python boundary.

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
    ...

def out_degree(row_offsets: Tensor, degrees: Tensor, vertex_count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute per-vertex out-degree from CSR row offsets.

    The graph descriptor uses uint32 row_offsets with length vertex_count + 1. The output degrees tensor has length vertex_count.

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
    ...

def in_degree(col_offsets: Tensor, degrees: Tensor, vertex_count: int, backend: str = "auto", *, queue: Queue | None = None, sync: bool = False) -> Tensor:
    """
    Compute per-vertex in-degree from CSC column offsets.

    The graph descriptor uses uint32 CSC/transposed-CSR col_offsets with length vertex_count + 1. The output degrees tensor has length vertex_count.

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
    ...