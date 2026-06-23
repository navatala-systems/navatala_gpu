// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

#include "gpu_library_ops.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <limits>
#include <memory>

namespace GpuRuntime {
namespace {

class UnsupportedLibraryOps : public LibraryOps {
public:
    // Stub builds intentionally report no BLAS/SparseLt capability so callers
    // fail loudly or select portable fallbacks instead of measuring a fake path.
    bool hasBlasSupport() const override { return false; }
    bool hasStructuredSparseSupport() const override { return false; }
    bool hasFftSupport() const override { return false; }
    bool hasSparseSupport() const override { return false; }
    bool hasSolverSupport() const override { return false; }
    bool hasSortReduceSupport() const override { return false; }

    LibraryStatus gemm(Queue&, const GemmParams&, Buffer&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus gemmBatched(
        Queue&, const GemmParams&, Buffer* const*, Buffer* const*, Buffer* const*,
        std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus gemmStridedBatched(
        Queue&, const GemmStridedParams&, Buffer&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus axpy(Queue&, const AxpyParams&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus dot(Queue&, const DotParams&, Buffer&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus nrm2(Queue&, const Nrm2Params&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus gemv(Queue&, const GemvParams&, Buffer&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus trsm(Queue&, const TrsmParams&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus syrk(Queue&, const SyrkParams&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus fft(Queue&, const FFTParams&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus fftInPlace(Queue&, const FFTParams&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus spMv(Queue&, const SpMVParams&, const SparseMatrixDescriptor&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus spMm(Queue&, const SpMMParams&, const SparseMatrixDescriptor&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus spGemm(
        Queue&, const SpGEMMParams&, const SparseMatrixDescriptor&,
        const SparseMatrixDescriptor&, SparseMatrixDescriptor&, std::int64_t&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus cholesky(Queue&, FillMode, Buffer&, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus lu(Queue&, Buffer&, Buffer&, std::int64_t, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus qr(Queue&, Buffer&, Buffer&, std::int64_t, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus svd(
        Queue&, Buffer&, Buffer&, Buffer&, Buffer&, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, bool, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus syev(Queue&, EigMode, FillMode, Buffer&, Buffer&, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus syevd(Queue&, EigMode, FillMode, Buffer&, Buffer&, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus geev(
        Queue&, EigMode, EigMode, Buffer&, Buffer&, Buffer&, Buffer&, Buffer&,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus csrToCoo(Queue&, Buffer&, Buffer&, std::int64_t, std::int64_t) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus cooToCsr(Queue&, Buffer&, Buffer&, std::int64_t, std::int64_t) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus csrToCsc(
        Queue&, Buffer&, Buffer&, Buffer&, Buffer&, Buffer&, Buffer&,
        std::int64_t, std::int64_t, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus cooSortByRow(Queue&, Buffer&, Buffer&, Buffer&, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus cooSortByCol(Queue&, Buffer&, Buffer&, Buffer&, std::int64_t, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus csrValidate(Queue&, Buffer&, Buffer&, std::int64_t, std::int64_t, std::int64_t, Buffer&) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus sortRadixPairsU32(
        Queue&, const SortPairsParams&, Buffer&, Buffer&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }

    LibraryStatus reduceByKeyU32(
        Queue&, const ReduceByKeyParams&, Buffer&, Buffer&, Buffer&, Buffer&, Buffer&, DataType) override
    {
        return LibraryStatus::NotSupported;
    }
};

rocblas_operation toRocBlasOp(TransposeOp op)
{
    switch (op) {
        case TransposeOp::NoTrans:
            return rocblas_operation_none;
        case TransposeOp::Trans:
            return rocblas_operation_transpose;
        case TransposeOp::ConjTrans:
            return rocblas_operation_conjugate_transpose;
    }
    return rocblas_operation_none;
}

bool fitsRocBlasInt(std::int64_t value)
{
    return value >= 0 && value <= static_cast<std::int64_t>(std::numeric_limits<rocblas_int>::max());
}

class HipRocBlasLibraryOps final : public UnsupportedLibraryOps {
public:
    HipRocBlasLibraryOps()
    {
        if (rocblas_create_handle(&handle_) != rocblas_status_success) {
            handle_ = nullptr;
        }
    }

    ~HipRocBlasLibraryOps() override
    {
        if (handle_) {
            rocblas_destroy_handle(handle_);
        }
    }

    bool hasBlasSupport() const override { return handle_ != nullptr; }

    LibraryStatus gemm(
        Queue& queue,
        const GemmParams& params,
        Buffer& A,
        Buffer& B,
        Buffer& C,
        DataType dtype) override
    {
        if (!handle_) {
            return LibraryStatus::NotInitialized;
        }
        if (dtype != DataType::Float32 && dtype != DataType::Float64) {
            return LibraryStatus::NotSupported;
        }
        if (!fitsRocBlasInt(params.m) || !fitsRocBlasInt(params.n) || !fitsRocBlasInt(params.k) ||
            !fitsRocBlasInt(params.lda) || !fitsRocBlasInt(params.ldb) || !fitsRocBlasInt(params.ldc)) {
            return LibraryStatus::InvalidValue;
        }

        if (rocblas_set_stream(handle_, static_cast<hipStream_t>(queue.nativeHandle())) != rocblas_status_success ||
            rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host) != rocblas_status_success) {
            return LibraryStatus::ExecutionFailed;
        }

        const auto transA = toRocBlasOp(params.transA);
        const auto transB = toRocBlasOp(params.transB);
        const rocblas_int m = static_cast<rocblas_int>(params.m);
        const rocblas_int n = static_cast<rocblas_int>(params.n);
        const rocblas_int k = static_cast<rocblas_int>(params.k);
        const rocblas_int lda = static_cast<rocblas_int>(params.lda);
        const rocblas_int ldb = static_cast<rocblas_int>(params.ldb);
        const rocblas_int ldc = static_cast<rocblas_int>(params.ldc);

        rocblas_status status = rocblas_status_internal_error;
        if (dtype == DataType::Float32) {
            const float alpha = static_cast<float>(params.alpha);
            const float beta = static_cast<float>(params.beta);
            status = rocblas_sgemm(
                handle_,
                transA,
                transB,
                m,
                n,
                k,
                &alpha,
                static_cast<const float*>(A.getDevicePointer()),
                lda,
                static_cast<const float*>(B.getDevicePointer()),
                ldb,
                &beta,
                static_cast<float*>(C.getDevicePointer()),
                ldc);
        } else {
            const double alpha = params.alpha;
            const double beta = params.beta;
            status = rocblas_dgemm(
                handle_,
                transA,
                transB,
                m,
                n,
                k,
                &alpha,
                static_cast<const double*>(A.getDevicePointer()),
                lda,
                static_cast<const double*>(B.getDevicePointer()),
                ldb,
                &beta,
                static_cast<double*>(C.getDevicePointer()),
                ldc);
        }

        return status == rocblas_status_success ? LibraryStatus::Success : LibraryStatus::ExecutionFailed;
    }

private:
    rocblas_handle handle_ = nullptr;
};

} // namespace

std::unique_ptr<LibraryOps> createHipLibraryOps()
{
    auto ops = std::make_unique<HipRocBlasLibraryOps>();
    if (!ops->hasBlasSupport()) {
        return nullptr;
    }
    return ops;
}

} // namespace GpuRuntime
