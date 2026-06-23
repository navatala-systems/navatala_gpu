// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
/**
 * GPU Library Operations Interface
 *
 * This header defines the abstract interface for GPU library operations
 * (cuBLAS/rocBLAS, cuFFT/rocFFT, cuSPARSE/rocSPARSE, cuSOLVER/rocSOLVER),
 * plus a small number of stdlib primitives (sort/reduce) needed for
 * deterministic scatter-reduce lowering.
 *
 * It is designed to be used together with the stable `gpu_runtime` API.
 *
 * Generated dispatch code (gpugen) includes this header when available.
 */

#include "gpu_runtime.h"

#include <cstdint>
#include <vector>
#include <memory>

namespace GpuRuntime {

// ============================================================================
// Enumerations
// ============================================================================

enum class LibraryStatus {
    Success = 0,
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
    Unknown = 999
};

enum class TransposeOp {
    NoTrans,
    Trans,
    ConjTrans
};

enum class SideMode {
    Left,
    Right
};

enum class FillMode {
    Upper,
    Lower
};

enum class DiagType {
    NonUnit,
    Unit
};

enum class SparseFormat {
    CSR,  // Compressed Sparse Row
    CSC,  // Compressed Sparse Column
    COO,  // Coordinate format
    ELL   // ELLPACK format
};

enum class StructuredSparsityKind {
    None,
    FiftyPercent,
    TwoOfFour
};

enum class StructuredSparseOperand {
    A,
    B
};

enum class StructuredSparseStorage {
    Dense,
    PrunedDense,
    Compressed
};

enum class StructuredSparseFallbackPolicy {
    Reject,
    DenseGemm,
    PortableKernel
};

enum class FFTType {
    C2C,  // Complex to Complex
    R2C,  // Real to Complex
    C2R   // Complex to Real
};

enum class FFTDirection {
    Forward,
    Inverse
};

enum class EigMode {
    NoVector,  // Compute eigenvalues only
    Vector     // Compute both eigenvalues and eigenvectors
};

// ============================================================================
// Deterministic Scatter-Reduce / StdLib Primitives
// ============================================================================

// Note: This intentionally mirrors the formal ReduceOp contract subset used by
// deterministic scatter-reduce lowering.
enum class ReduceOp {
    Sum,
    Min,
    Max
};

// Controls for device-side reduction numerics (Tier-C P2).
// Default keeps vendor/library semantics. FP64Accumulate accumulates Float32 sums in Float64
// (then casts back) to improve reproducibility/accuracy for long reductions.
enum class ReduceNumericMode {
    Default = 0,
    FP64Accumulate = 1,
    // Deterministic (fixed-order) reduction intended for debugging/regression.
    // Backends may implement this as a slower multi-pass reduction.
    DeterministicTree = 2,
    // Compensated summation (primarily useful for host fallbacks and debugging).
    Kahan = 3,
    // Pairwise (tree-structured) summation to reduce error vs naive left-fold.
    Pairwise = 4
};

struct SortPairsParams {
    std::uint32_t n = 0;  // number of pairs
    // When true, sort keys in descending order (larger keys first).
    // Backends may return NotSupported; generated code will fall back to host sort.
    bool descending = false;
};

struct SortKeysParams {
    std::uint32_t n = 0;  // number of keys
    // When true, sort in descending order (larger keys first).
    bool descending = false;
};

struct ReduceByKeyParams {
    ReduceOp op = ReduceOp::Sum;
    std::uint32_t n = 0;  // number of input pairs (sorted)
    ReduceNumericMode numericMode = ReduceNumericMode::Default;
};

// Buffer/Queue/Device are defined by `gpu_runtime.h` and imported above.

// ============================================================================
// BLAS Parameter Structures
// ============================================================================

struct GemmParams {
    TransposeOp transA = TransposeOp::NoTrans;
    TransposeOp transB = TransposeOp::NoTrans;
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::int64_t k = 0;
    double alpha = 1.0;
    double beta = 0.0;
    std::int64_t lda = 0;
    std::int64_t ldb = 0;
    std::int64_t ldc = 0;
};

struct GemmStridedParams : GemmParams {
    std::int64_t strideA = 0;
    std::int64_t strideB = 0;
    std::int64_t strideC = 0;
    std::int64_t batchCount = 1;
};

struct AxpyParams {
    std::int64_t n = 0;
    double alpha = 1.0;
    std::int64_t incx = 1;
    std::int64_t incy = 1;
};

struct DotParams {
    std::int64_t n = 0;
    std::int64_t incx = 1;
    std::int64_t incy = 1;
    ReduceNumericMode numericMode = ReduceNumericMode::Default;
};

struct Nrm2Params {
    std::int64_t n = 0;
    std::int64_t incx = 1;
    ReduceNumericMode numericMode = ReduceNumericMode::Default;
};

struct GemvParams {
    TransposeOp trans = TransposeOp::NoTrans;
    std::int64_t m = 0;
    std::int64_t n = 0;
    double alpha = 1.0;
    double beta = 0.0;
    std::int64_t lda = 0;
    std::int64_t incx = 1;
    std::int64_t incy = 1;
};

struct TrsmParams {
    SideMode side = SideMode::Left;
    FillMode uplo = FillMode::Upper;
    TransposeOp trans = TransposeOp::NoTrans;
    DiagType diag = DiagType::NonUnit;
    std::int64_t m = 0;
    std::int64_t n = 0;
    double alpha = 1.0;
    std::int64_t lda = 0;
    std::int64_t ldb = 0;
};

struct SyrkParams {
    FillMode uplo = FillMode::Upper;
    TransposeOp trans = TransposeOp::NoTrans;
    std::int64_t n = 0;
    std::int64_t k = 0;
    double alpha = 1.0;
    double beta = 0.0;
    std::int64_t lda = 0;
    std::int64_t ldc = 0;
};

// ============================================================================
// FFT Parameter Structure
// ============================================================================

struct FFTParams {
    FFTType type = FFTType::C2C;
    FFTDirection direction = FFTDirection::Forward;
    std::vector<std::int64_t> dims;
    std::int64_t batchCount = 1;
};

// ============================================================================
// Sparse Matrix Structures
// ============================================================================

struct SparseMatrixDescriptor {
    SparseFormat format = SparseFormat::CSR;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t nnz = 0;

    // CSR format: rowPtr[rows+1], colInd[nnz], values[nnz]
    Buffer* rowPtr = nullptr;
    Buffer* colInd = nullptr;
    Buffer* values = nullptr;

    // COO format: rowInd[nnz], colInd[nnz], values[nnz]
    Buffer* rowInd = nullptr;

    // CSC format: colPtr[cols+1], rowInd[nnz], values[nnz]
    Buffer* colPtr = nullptr;

    // ELL format: colInd[rows*ellWidth], values[rows*ellWidth]
    std::int64_t ellWidth = 0;
};

struct SpMVParams {
    SparseFormat format = SparseFormat::CSR;
    TransposeOp transA = TransposeOp::NoTrans;
    std::int64_t m = 0;      // Number of rows in sparse matrix A
    std::int64_t n = 0;      // Number of columns in sparse matrix A
    std::int64_t nnz = 0;    // Number of non-zeros in A
    std::int64_t ellWidth = 0;  // ELL format: max non-zeros per row
    double alpha = 1.0;
    double beta = 0.0;
};

struct SpMMParams {
    SparseFormat format = SparseFormat::CSR;
    TransposeOp transA = TransposeOp::NoTrans;
    TransposeOp transB = TransposeOp::NoTrans;
    std::int64_t m = 0;      // Rows of A and C
    std::int64_t n = 0;      // Columns of B and C
    std::int64_t k = 0;      // Columns of A, rows of B
    std::int64_t nnz = 0;    // Non-zeros in A
    std::int64_t ellWidth = 0;  // ELL format: max non-zeros per row
    std::int64_t ldb = 0;    // Leading dimension of B
    std::int64_t ldc = 0;    // Leading dimension of C
    double alpha = 1.0;
    double beta = 0.0;
};

struct SpGEMMParams {
    SparseFormat format = SparseFormat::CSR;
    TransposeOp transA = TransposeOp::NoTrans;
    TransposeOp transB = TransposeOp::NoTrans;
    std::int64_t m = 0;      // Rows of A and C
    std::int64_t n = 0;      // Columns of B and C
    std::int64_t k = 0;      // Columns of A, rows of B
    std::int64_t nnzA = 0;
    std::int64_t nnzB = 0;
};

struct StructuredSparseMatmulParams {
    TransposeOp transA = TransposeOp::NoTrans;
    TransposeOp transB = TransposeOp::NoTrans;
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::int64_t k = 0;
    double alpha = 1.0;
    double beta = 0.0;
    std::int64_t lda = 0;
    std::int64_t ldb = 0;
    std::int64_t ldc = 0;
    StructuredSparsityKind sparsity = StructuredSparsityKind::TwoOfFour;
    StructuredSparseOperand sparseOperand = StructuredSparseOperand::A;
    StructuredSparseStorage storage = StructuredSparseStorage::PrunedDense;
    StructuredSparseFallbackPolicy fallbackPolicy = StructuredSparseFallbackPolicy::Reject;
    bool allowRuntimePrune = false;
    bool allowRuntimeCompress = true;
};

// ============================================================================
// Linear Solve / Two-Level Preconditioner Types
// ============================================================================

enum class LinearSolveBackend {
    Native,
    Ginkgo,
    AmgX
};

enum class LinearSolvePreconditioner {
    None,
    Jacobi,
    Chebyshev,
    TwoLevel,
    AMG
};

enum class LinearSolveReusePolicy {
    RebuildAlways,
    FreezeAfterSetup,
    RefreshNumericOnly
};

enum class LinearSolveCoarseBackend {
    Host,
    Device,
    Delegated
};

using LinearSolveHandle = std::uint64_t;
using LinearSolveHierarchyHandle = std::uint64_t;

struct LinearSolveParams {
    LinearSolveBackend backend = LinearSolveBackend::Native;
    LinearSolvePreconditioner preconditioner = LinearSolvePreconditioner::Jacobi;
    std::int64_t maxIterations = 100;
    double tolerance = 1.0e-6;
    double relativeTolerance = 0.0;
    LinearSolveReusePolicy reusePolicy = LinearSolveReusePolicy::RefreshNumericOnly;
};

struct TwoLevelParams {
    std::int64_t smootherSweeps = 2;
    double chebyshevAlpha = 0.67;
    double chebyshevBeta = 0.33;
    LinearSolveCoarseBackend coarseBackend = LinearSolveCoarseBackend::Host;
    std::int64_t refreshEveryCorrections = 1;
    std::int64_t freezeStructureAfterStep = -1;
};

struct LinearSolveStats {
    std::int64_t iterations = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
    double solveSeconds = 0.0;
};

// ============================================================================
// NCCL Types
// ============================================================================

/** Reduction operations for NCCL collectives. */
enum class NcclRedOp {
    Sum,
    Prod,
    Min,
    Max,
    Avg
};

/** Opaque NCCL communicator handle. */
using NcclComm = void*;

// ============================================================================
// LibraryOps - Abstract interface for GPU library operations
// ============================================================================

class LibraryOps {
public:
    virtual ~LibraryOps() = default;

    // =========================================================================
    // Capability queries
    // =========================================================================

    virtual bool hasBlasSupport() const = 0;
    virtual bool hasFftSupport() const = 0;
    virtual bool hasSparseSupport() const = 0;
    virtual bool hasStructuredSparseSupport() const { return false; }
    virtual bool hasSolverSupport() const = 0;
    virtual bool hasSortReduceSupport() const = 0;
    virtual bool hasNcclSupport() const { return false; }
    virtual bool hasLinearSolveSupport() const { return false; }

    // =========================================================================
    // BLAS Operations
    // =========================================================================

    /** General matrix multiplication: C = alpha * op(A) * op(B) + beta * C */
    virtual LibraryStatus gemm(
        Queue& queue,
        const GemmParams& params,
        Buffer& A,
        Buffer& B,
        Buffer& C,
        DataType dtype) = 0;

    /** Batched GEMM with array of matrices */
    virtual LibraryStatus gemmBatched(
        Queue& queue,
        const GemmParams& params,
        Buffer* const* Aarray,
        Buffer* const* Barray,
        Buffer* const* Carray,
        std::int64_t batchCount,
        DataType dtype) = 0;

    /** Strided batched GEMM */
    virtual LibraryStatus gemmStridedBatched(
        Queue& queue,
        const GemmStridedParams& params,
        Buffer& A,
        Buffer& B,
        Buffer& C,
        DataType dtype) = 0;

    /** AXPY: y = alpha * x + y */
    virtual LibraryStatus axpy(
        Queue& queue,
        const AxpyParams& params,
        Buffer& x,
        Buffer& y,
        DataType dtype) = 0;

    /** Dot product: result = x^T * y */
    virtual LibraryStatus dot(
        Queue& queue,
        const DotParams& params,
        Buffer& x,
        Buffer& y,
        Buffer& result,
        DataType dtype) = 0;

    /** Euclidean norm: result = ||x||_2 */
    virtual LibraryStatus nrm2(
        Queue& queue,
        const Nrm2Params& params,
        Buffer& x,
        Buffer& result,
        DataType dtype) = 0;

    /** Matrix-vector multiplication: y = alpha * op(A) * x + beta * y */
    virtual LibraryStatus gemv(
        Queue& queue,
        const GemvParams& params,
        Buffer& A,
        Buffer& x,
        Buffer& y,
        DataType dtype) = 0;

    /** Triangular solve: op(A) * X = alpha * B  or  X * op(A) = alpha * B */
    virtual LibraryStatus trsm(
        Queue& queue,
        const TrsmParams& params,
        Buffer& A,
        Buffer& B,
        DataType dtype) = 0;

    /** Symmetric rank-k update: C = alpha * op(A) * op(A)^T + beta * C */
    virtual LibraryStatus syrk(
        Queue& queue,
        const SyrkParams& params,
        Buffer& A,
        Buffer& C,
        DataType dtype) = 0;

    // =========================================================================
    // FFT Operations
    // =========================================================================

    /** FFT: out-of-place transform */
    virtual LibraryStatus fft(
        Queue& queue,
        const FFTParams& params,
        Buffer& input,
        Buffer& output,
        DataType dtype) = 0;

    /** FFT: in-place transform */
    virtual LibraryStatus fftInPlace(
        Queue& queue,
        const FFTParams& params,
        Buffer& data,
        DataType dtype) = 0;

    // =========================================================================
    // Sparse Operations
    // =========================================================================

    /** Sparse matrix-vector multiplication: y = alpha * op(A) * x + beta * y */
    virtual LibraryStatus spMv(
        Queue& queue,
        const SpMVParams& params,
        const SparseMatrixDescriptor& A,
        Buffer& x,
        Buffer& y,
        DataType dtype) = 0;

    /** Sparse matrix-dense matrix multiplication: C = alpha * op(A) * op(B) + beta * C */
    virtual LibraryStatus spMm(
        Queue& queue,
        const SpMMParams& params,
        const SparseMatrixDescriptor& A,
        Buffer& B,
        Buffer& C,
        DataType dtype) = 0;

    /** Sparse-sparse matrix multiplication: C = op(A) * op(B) */
    virtual LibraryStatus spGemm(
        Queue& queue,
        const SpGEMMParams& params,
        const SparseMatrixDescriptor& A,
        const SparseMatrixDescriptor& B,
        SparseMatrixDescriptor& C,
        std::int64_t& nnzC,
        DataType dtype) = 0;

    /** Structured sparse matrix multiplication:
        C = alpha * op(A) * op(B) + beta * C where one operand has a declared
        50% / 2:4 structured sparsity pattern. This is for hipSPARSELt /
        cuSPARSELt style paths, not arbitrary CSR/COO sparse matrices. */
    virtual LibraryStatus structuredSparseMatmul(
        Queue& queue,
        const StructuredSparseMatmulParams& params,
        Buffer& A,
        Buffer& B,
        Buffer& C,
        DataType dtype) {
        (void)queue;
        (void)params;
        (void)A;
        (void)B;
        (void)C;
        (void)dtype;
        return LibraryStatus::NotSupported;
    }

    // =========================================================================
    // Linear Solve / Two-Level Preconditioner Operations
    // =========================================================================

    /** Create a backend-specific linear-solve context.
        The returned handle is opaque to callers. */
    virtual LibraryStatus createLinearSolveContext(
        LinearSolveBackend backend,
        LinearSolveHandle& outHandle) {
        (void)backend;
        (void)outHandle;
        return LibraryStatus::NotSupported;
    }

    /** Destroy a backend-specific linear-solve context. */
    virtual LibraryStatus destroyLinearSolveContext(
        LinearSolveHandle handle) {
        (void)handle;
        return LibraryStatus::NotSupported;
    }

    /** Build or initialize a hierarchy for a two-level / AMG-style preconditioner. */
    virtual LibraryStatus createLinearSolveHierarchy(
        Queue& queue,
        LinearSolveHandle solver,
        const SparseMatrixDescriptor& A,
        const LinearSolveParams& solveParams,
        const TwoLevelParams& twoLevelParams,
        LinearSolveHierarchyHandle& outHierarchy,
        DataType dtype) {
        (void)queue;
        (void)solver;
        (void)A;
        (void)solveParams;
        (void)twoLevelParams;
        (void)outHierarchy;
        (void)dtype;
        return LibraryStatus::NotSupported;
    }

    /** Refresh numeric hierarchy data while reusing previously built structure. */
    virtual LibraryStatus refreshLinearSolveHierarchy(
        Queue& queue,
        LinearSolveHandle solver,
        LinearSolveHierarchyHandle hierarchy,
        const SparseMatrixDescriptor& A,
        DataType dtype) {
        (void)queue;
        (void)solver;
        (void)hierarchy;
        (void)A;
        (void)dtype;
        return LibraryStatus::NotSupported;
    }

    /** Apply the configured preconditioner or hierarchy to `rhs`, writing into `out`. */
    virtual LibraryStatus applyLinearSolvePreconditioner(
        Queue& queue,
        LinearSolveHandle solver,
        LinearSolveHierarchyHandle hierarchy,
        const LinearSolveParams& solveParams,
        const TwoLevelParams& twoLevelParams,
        Buffer& rhs,
        Buffer& out,
        DataType dtype,
        LinearSolveStats* stats = nullptr) {
        (void)queue;
        (void)solver;
        (void)hierarchy;
        (void)solveParams;
        (void)twoLevelParams;
        (void)rhs;
        (void)out;
        (void)dtype;
        (void)stats;
        return LibraryStatus::NotSupported;
    }

    /** Destroy a previously created linear-solve hierarchy. */
    virtual LibraryStatus destroyLinearSolveHierarchy(
        LinearSolveHandle solver,
        LinearSolveHierarchyHandle hierarchy) {
        (void)solver;
        (void)hierarchy;
        return LibraryStatus::NotSupported;
    }

    // =========================================================================
    // Solver Operations
    // =========================================================================

    /** Cholesky factorization: A = L * L^T (or U^T * U) */
    virtual LibraryStatus cholesky(
        Queue& queue,
        FillMode uplo,
        Buffer& A,
        std::int64_t n,
        std::int64_t lda,
        DataType dtype) = 0;

    /** LU factorization with partial pivoting: P * A = L * U */
    virtual LibraryStatus lu(
        Queue& queue,
        Buffer& A,
        Buffer& ipiv,
        std::int64_t m,
        std::int64_t n,
        std::int64_t lda,
        DataType dtype) = 0;

    /** QR factorization: A = Q * R */
    virtual LibraryStatus qr(
        Queue& queue,
        Buffer& A,
        Buffer& tau,
        std::int64_t m,
        std::int64_t n,
        std::int64_t lda,
        DataType dtype) = 0;

    /** Singular Value Decomposition: A = U * S * V^T */
    virtual LibraryStatus svd(
        Queue& queue,
        Buffer& A,
        Buffer& U,
        Buffer& S,
        Buffer& Vt,
        std::int64_t m,
        std::int64_t n,
        std::int64_t lda,
        std::int64_t ldu,
        std::int64_t ldvt,
        bool fullMatrices,
        DataType dtype) = 0;

    /** Symmetric eigenvalue decomposition (QR algorithm): A = V * diag(W) * V^T
        A is overwritten with eigenvectors V when eigMode == Vector.
        W contains the eigenvalues in ascending order. */
    virtual LibraryStatus syev(
        Queue& queue,
        EigMode eigMode,
        FillMode uplo,
        Buffer& A,
        Buffer& W,
        std::int64_t n,
        std::int64_t lda,
        DataType dtype) = 0;

    /** Symmetric eigenvalue decomposition (divide-and-conquer, faster for large matrices).
        Same interface as syev but uses a different algorithm internally. */
    virtual LibraryStatus syevd(
        Queue& queue,
        EigMode eigMode,
        FillMode uplo,
        Buffer& A,
        Buffer& W,
        std::int64_t n,
        std::int64_t lda,
        DataType dtype) = 0;

    /** General (non-symmetric) eigenvalue decomposition.
        Computes eigenvalues and optionally left/right eigenvectors.
        Eigenvalues are returned as (WR, WI) pairs for real/imaginary parts.
        For real matrices, complex eigenvalues come in conjugate pairs. */
    virtual LibraryStatus geev(
        Queue& queue,
        EigMode leftEigMode,
        EigMode rightEigMode,
        Buffer& A,
        Buffer& WR,
        Buffer& WI,
        Buffer& VL,
        Buffer& VR,
        std::int64_t n,
        std::int64_t lda,
        std::int64_t ldvl,
        std::int64_t ldvr,
        DataType dtype) = 0;

    // =========================================================================
    // Sparse Format Conversion Operations
    // =========================================================================

    /** CSR to COO conversion: expands rowPtr to rowInd.
        rowPtr: input CSR row pointer array [m+1]
        rowIndOut: output COO row index array [nnz] */
    virtual LibraryStatus csrToCoo(
        Queue& queue,
        Buffer& rowPtr,
        Buffer& rowIndOut,
        std::int64_t m,
        std::int64_t nnz) = 0;

    /** COO to CSR conversion: compresses rowInd to rowPtr.
        rowInd: input COO row index array [nnz]
        rowPtrOut: output CSR row pointer array [m+1] */
    virtual LibraryStatus cooToCsr(
        Queue& queue,
        Buffer& rowInd,
        Buffer& rowPtrOut,
        std::int64_t m,
        std::int64_t nnz) = 0;

    /** CSR to CSC conversion: transposes sparse structure.
        Converts from CSR(rowPtr, colInd, values) to CSC(colPtrOut, rowIndOut, valuesOut). */
    virtual LibraryStatus csrToCsc(
        Queue& queue,
        Buffer& rowPtr,
        Buffer& colInd,
        Buffer& values,
        Buffer& colPtrOut,
        Buffer& rowIndOut,
        Buffer& valuesOut,
        std::int64_t m,
        std::int64_t n,
        std::int64_t nnz,
        DataType dtype) = 0;

    /** Sort COO matrix by row indices (primary), then column indices (secondary).
        In-place sorting of (rowInd, colInd, values) arrays. */
    virtual LibraryStatus cooSortByRow(
        Queue& queue,
        Buffer& rowInd,
        Buffer& colInd,
        Buffer& values,
        std::int64_t nnz,
        DataType dtype) = 0;

    /** Sort COO matrix by column indices (primary), then row indices (secondary).
        In-place sorting of (rowInd, colInd, values) arrays. */
    virtual LibraryStatus cooSortByCol(
        Queue& queue,
        Buffer& rowInd,
        Buffer& colInd,
        Buffer& values,
        std::int64_t nnz,
        DataType dtype) = 0;

    /** Validate CSR structure integrity.
        Checks: sorted column indices within each row, valid index ranges.
        info: output buffer for validation result (0 = valid, >0 = first invalid row). */
    virtual LibraryStatus csrValidate(
        Queue& queue,
        Buffer& rowPtr,
        Buffer& colInd,
        std::int64_t m,
        std::int64_t n,
        std::int64_t nnz,
        Buffer& info) = 0;

    // =========================================================================
    // Sort / Reduce (stdlib primitives)
    // =========================================================================

    /** Stable radix sort of (key,value) pairs by key.
        Keys are always UInt32 (v1). Values are typed by `dtype`.
        Inputs and outputs may alias only if the backend supports in-place sort (not required). */
    virtual LibraryStatus sortRadixPairsU32(
        Queue& queue,
        const SortPairsParams& params,
        Buffer& keysIn,
        Buffer& valsIn,
        Buffer& keysOut,
        Buffer& valsOut,
        DataType dtype) = 0;

    /** Stable radix sort of keys (UInt32).
        Descending order is controlled via `params.descending`. */
    virtual LibraryStatus sortRadixKeysU32(
        Queue& queue,
        const SortKeysParams& params,
        Buffer& keysIn,
        Buffer& keysOut) {
        (void)queue;
        (void)params;
        (void)keysIn;
        (void)keysOut;
        return LibraryStatus::NotSupported;
    }

    /** Stable radix sort of keys (Float32).
        Descending order is controlled via `params.descending`. */
    virtual LibraryStatus sortRadixKeysF32(
        Queue& queue,
        const SortKeysParams& params,
        Buffer& keysIn,
        Buffer& keysOut) {
        (void)queue;
        (void)params;
        (void)keysIn;
        (void)keysOut;
        return LibraryStatus::NotSupported;
    }

    /** Reduce-by-key for sorted (key,value) pairs.
        Computes unique keys and reduced values; writes `outCount[0]` (#unique keys). */
    virtual LibraryStatus reduceByKeyU32(
        Queue& queue,
        const ReduceByKeyParams& params,
        Buffer& keysIn,
        Buffer& valsIn,
        Buffer& keysOut,
        Buffer& valsOut,
        Buffer& outCount,
        DataType dtype) = 0;

    // =========================================================================
    // NCCL / RCCL Operations
    // =========================================================================

    /** AllReduce: recvbuf[i] = op(sendbuf[i]) across all ranks */
    virtual LibraryStatus ncclAllReduce(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t count, DataType datatype, NcclRedOp op,
        NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)count;
        (void)datatype; (void)op; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** ReduceScatter: each rank receives reduced chunk */
    virtual LibraryStatus ncclReduceScatter(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t recvcount, DataType datatype, NcclRedOp op,
        NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)recvcount;
        (void)datatype; (void)op; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** Reduce: reduce to root rank */
    virtual LibraryStatus ncclReduce(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t count, DataType datatype, NcclRedOp op,
        int root, NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)count;
        (void)datatype; (void)op; (void)root; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** AllGather: gather sendcount elements from each rank */
    virtual LibraryStatus ncclAllGather(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t sendcount, DataType datatype,
        NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)sendcount;
        (void)datatype; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** Broadcast: broadcast count elements from root */
    virtual LibraryStatus ncclBroadcast(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t count, DataType datatype,
        int root, NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)count;
        (void)datatype; (void)root; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** Send: point-to-point send to peer */
    virtual LibraryStatus ncclSend(
        Queue& queue, const void* sendbuf,
        size_t count, DataType datatype,
        int peer, NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)count;
        (void)datatype; (void)peer; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** Recv: point-to-point receive from peer */
    virtual LibraryStatus ncclRecv(
        Queue& queue, void* recvbuf,
        size_t count, DataType datatype,
        int peer, NcclComm comm, void* stream) {
        (void)queue; (void)recvbuf; (void)count;
        (void)datatype; (void)peer; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** AllToAll: each rank sends/receives count elements to/from every other rank.
        Returns NotSupported if NCCL version is too old. */
    virtual LibraryStatus ncclAllToAll(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t count, DataType datatype,
        NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)count;
        (void)datatype; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** Scatter: root scatters count elements to each rank.
        Returns NotSupported if NCCL version is too old. */
    virtual LibraryStatus ncclScatter(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t count, DataType datatype,
        int root, NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)count;
        (void)datatype; (void)root; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }

    /** Gather: root gathers count elements from each rank.
        Returns NotSupported if NCCL version is too old. */
    virtual LibraryStatus ncclGather(
        Queue& queue, const void* sendbuf, void* recvbuf,
        size_t count, DataType datatype,
        int root, NcclComm comm, void* stream) {
        (void)queue; (void)sendbuf; (void)recvbuf; (void)count;
        (void)datatype; (void)root; (void)comm; (void)stream;
        return LibraryStatus::NotSupported;
    }
};

// ============================================================================
// Factory function declarations (implemented in platform-specific files)
// ============================================================================

/**
 * Create a LibraryOps instance for the current platform.
 * Returns nullptr if no GPU libraries are available.
 */
std::unique_ptr<LibraryOps> createLibraryOps();

/**
 * Create a LibraryOps instance matching a gpugen backend string ("cuda", "hip", "metal").
 * Returns nullptr if no matching GPU libraries are available.
 */
std::unique_ptr<LibraryOps> createLibraryOpsForBackend(const std::string& backend);

/**
 * Create a LibraryOps instance for a specific platform.
 */
std::unique_ptr<LibraryOps> createCudaLibraryOps();
std::unique_ptr<LibraryOps> createHipLibraryOps();
std::unique_ptr<LibraryOps> createMetalLibraryOps();
std::unique_ptr<LibraryOps> createGinkgoLibraryOps();
std::unique_ptr<LibraryOps> createAmgXLibraryOps();

} // namespace GpuRuntime
