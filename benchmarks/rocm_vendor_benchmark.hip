// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// ROCm vendor comparison benchmark.
//
// This benchmark intentionally calls a small set of Navatala HIP kernels
// directly and compares them with ROCm vendor libraries. It is not part of the
// normal build; enable it with NAVATALA_GPU_BUILD_BENCHMARKS=ON on a ROCm host.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>
#include <navatala/navatala_ffi.h>
#if NAVATALA_HAVE_HIPSPARSELT
#include <hipsparselt/hipsparselt.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../kernels/hip/sparse/axpy_f32.hip"
#include "../kernels/hip/graph/spmv_weighted_f32.hip"
#include "../kernels/hip/graph/spmv_weighted_subgroup_f32.hip"
#include "../kernels/hip/transformer/tiled_gemm_f32.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_f32_out.hip"
#if NAVATALA_HAVE_MFMA_BENCHMARK
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_k_loop.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_direct.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_early_barrier.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_padded.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_pipelined.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_edge.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_edge_nn.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_edge_region.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_semantic.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta128.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta128_edge.hip"
#include "../kernels/hip/transformer/tiled_gemm_f16_mfma_cta128_semantic.hip"
#endif

namespace {

struct Options {
    int iterations = 50;
    int warmup = 10;
    bool quick = false;
    bool listOnly = false;
    bool requireHipSparseLt = false;
    bool includeHipSparseLtBenchmark = false;
    bool includeWrapperGemmBenchmark = false;
    bool includeMfmaBenchmark = false;
    std::string matrix = "standard";
    std::string jsonPath;
    std::string csrFixtureJsonPath;
    std::string csrFixtureBinPath;
};

struct CsrFixture {
    std::string jsonPath;
    std::string binPath;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint32_t indexBase = 0;
    std::vector<std::uint32_t> rowPtr;
    std::vector<std::uint32_t> colIdx;
    std::vector<float> values;
    std::vector<float> x;
};

struct RowDistribution {
    std::uint32_t minNnz = 0;
    double meanNnz = 0.0;
    std::array<std::uint32_t, 11> decileNnz{};
    std::uint32_t p50Nnz = 0;
    std::uint32_t p90Nnz = 0;
    std::uint32_t p99Nnz = 0;
    std::uint32_t maxNnz = 0;
    std::uint64_t threadRows = 0;
    std::uint64_t subgroupRows = 0;
};

struct Result {
    std::string operation;
    std::string shape;
    std::string navatalaPath;
    std::string vendorPath;
    bool correctness = false;
    double navatalaMs = 0.0;
    double vendorMs = 0.0;
    double generatedOverVendorRatio = 0.0;
    double maxAbsError = 0.0;
    std::string notes;
    std::string kernelClass = "scalar";
    std::string implementationKind = "portable_kernel";
    std::string tuningPath;
    std::uint32_t spmvRowNnzThreshold = 0;
    bool vendorDispatchSelected = false;
    double hostSubmitMeanMs = 0.0;
    double hostWallMeanMs = 0.0;
};

struct StreamTiming {
    double eventMeanMs = 0.0;
    double hostSubmitMeanMs = 0.0;
    double hostWallMeanMs = 0.0;
};

[[noreturn]] void die(const char* label, hipError_t status) {
    std::fprintf(stderr, "ERROR: %s failed: %s\n", label, hipGetErrorString(status));
    std::exit(2);
}

[[noreturn]] void die(const char* label, rocblas_status status) {
    std::fprintf(stderr, "ERROR: %s failed: rocblas_status=%d\n", label, static_cast<int>(status));
    std::exit(2);
}

[[noreturn]] void die(const char* label, rocsparse_status status) {
    std::fprintf(stderr, "ERROR: %s failed: rocsparse_status=%d\n", label, static_cast<int>(status));
    std::exit(2);
}

[[noreturn]] void die(const char* label, NavatalaErrorCode status) {
    std::fprintf(stderr, "ERROR: %s failed: NavatalaErrorCode=%d\n", label, static_cast<int>(status));
    std::exit(2);
}

void check(const char* label, hipError_t status) {
    if (status != hipSuccess) die(label, status);
}

void check(const char* label, rocblas_status status) {
    if (status != rocblas_status_success) die(label, status);
}

void check(const char* label, rocsparse_status status) {
    if (status != rocsparse_status_success) die(label, status);
}

void check(const char* label, NavatalaErrorCode status) {
    if (status != NAVATALA_SUCCESS) die(label, status);
}

void releaseHip(void* ptr) {
    if (ptr != nullptr) {
        (void)hipFree(ptr);
    }
}

#if NAVATALA_HAVE_HIPSPARSELT
[[noreturn]] void dieHipSparseLt(const char* label, hipsparseStatus_t status) {
    std::fprintf(stderr, "ERROR: %s failed: hipsparseStatus_t=%d\n", label, static_cast<int>(status));
    std::exit(2);
}

void checkHipSparseLt(const char* label, hipsparseStatus_t status) {
    if (status != HIPSPARSE_STATUS_SUCCESS) dieHipSparseLt(label, status);
}
#endif

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            options.quick = true;
            options.iterations = 10;
            options.warmup = 3;
        } else if (arg == "--list") {
            options.listOnly = true;
        } else if (arg == "--require-hipsparselt") {
            options.requireHipSparseLt = true;
        } else if (arg == "--include-hipsparselt-benchmark") {
            options.includeHipSparseLtBenchmark = true;
        } else if (arg == "--include-wrapper-gemm-benchmark") {
            options.includeWrapperGemmBenchmark = true;
        } else if (arg == "--include-mfma-benchmark") {
            options.includeMfmaBenchmark = true;
        } else if (arg == "--matrix" && i + 1 < argc) {
            options.matrix = argv[++i];
            if (options.matrix != "standard" && options.matrix != "broad" &&
                options.matrix != "cta128-evidence" &&
                options.matrix != "edge-tails" &&
                options.matrix != "wrapper-semantics" &&
                options.matrix != "cfd-fixture") {
                std::fprintf(stderr, "ERROR: --matrix must be 'standard', 'broad', 'cta128-evidence', 'edge-tails', 'wrapper-semantics', or 'cfd-fixture'\n");
                std::exit(2);
            }
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--json" && i + 1 < argc) {
            options.jsonPath = argv[++i];
        } else if (arg == "--csr-fixture-json" && i + 1 < argc) {
            options.csrFixtureJsonPath = argv[++i];
        } else if (arg == "--csr-fixture-bin" && i + 1 < argc) {
            options.csrFixtureBinPath = argv[++i];
        } else {
            std::fprintf(stderr,
                         "usage: rocm_vendor_benchmark [--quick] [--list] "
                         "[--require-hipsparselt] [--include-hipsparselt-benchmark] "
                         "[--include-wrapper-gemm-benchmark] "
                         "[--include-mfma-benchmark] "
                         "[--matrix standard|broad|cta128-evidence|edge-tails|wrapper-semantics|cfd-fixture] [--iterations N] "
                         "[--warmup N] [--json path] "
                         "[--csr-fixture-json path] [--csr-fixture-bin path]\n");
            std::exit(2);
        }
    }
    if (options.matrix == "cfd-fixture" && options.csrFixtureBinPath.empty()) {
        std::fprintf(stderr, "ERROR: --matrix cfd-fixture requires --csr-fixture-bin\n");
        std::exit(2);
    }
    return options;
}

void printBenchmarkList() {
    std::printf("ROCm vendor benchmark operations:\n");
    std::printf("  AXPY_F32: generated navatala_sparse_axpy_f32 vs rocBLAS rocblas_saxpy\n");
    std::printf("  GEMM_F32: generated navatala_transformer_tiled_gemm_f32 vs rocBLAS rocblas_sgemm\n");
    std::printf("  GEMM_F16_PORTABLE_F32OUT: generated navatala_transformer_tiled_gemm_f16_f32_out vs rocBLAS F16 input/F32 output rocblas_gemm_ex\n");
    std::printf("  GEMM_F32_WRAPPER_VENDOR: public C ABI navatala_gpu_gemm_f32 vendor dispatch vs rocBLAS rocblas_sgemm (opt-in)\n");
    std::printf("  GEMM_F16_MFMA: generated navatala_transformer_tiled_gemm_f16_mfma_k_loop vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+)\n");
    std::printf("  GEMM_F16_MFMA_CTA64_DIRECT: generated navatala_transformer_tiled_gemm_f16_mfma_cta64_direct vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, low-pressure direct-load row)\n");
    std::printf("  GEMM_F16_MFMA_CTA64_SHARED: generated navatala_transformer_tiled_gemm_f16_mfma_cta64_shared vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, low-pressure LDS-staged row)\n");
    std::printf("  GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER: generated navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, LDS staged with early release barrier)\n");
    std::printf("  GEMM_F16_MFMA_CTA64_SHARED_PADDED: generated navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, low-pressure padded LDS row)\n");
    std::printf("  GEMM_F16_MFMA_CTA64_PIPELINED: generated navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, R6 two-slot staged-panel row)\n");
    std::printf("  GEMM_F16_MFMA_CTA128: generated navatala_transformer_tiled_gemm_f16_mfma_cta128 vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, tile-divisible only)\n");
    std::printf("  GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN: generated navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, NN alpha=1 beta=0 true-tail candidate)\n");
    std::printf("  GEMM_F16_MFMA_CTA{64,128}_SEMANTIC_{ALPHA_BETA,TRANSPOSE}: direct semantic full-tile MFMA rows vs rocBLAS F16/F32 reference (opt-in, gfx942/ROCm 6.2+)\n");
    std::printf("  GEMM_F16_F32_WRAPPER_MFMA: public C ABI navatala_gpu_gemm_f16_f32 MFMA dispatch vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, edge-capable wrapper path)\n");
    std::printf("  GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT: public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_MFMA_MODE=split vs rocBLAS F16/F32 rocblas_gemm_ex (opt-in, gfx942/ROCm 6.2+, structural edge split path)\n");
    std::printf("  GEMM_F16_F32_WRAPPER_MFMA_{ALPHA_BETA,TRANSPOSE,BATCHED}: public C ABI MFMA semantic rows vs rocBLAS F16/F32 reference (opt-in, gfx942/ROCm 6.2+)\n");
    std::printf("  GEMM_F16_F32_WRAPPER_{AUTO,VENDOR}_{TAIL_LARGE,TRANSPOSE,BATCHED}: public C ABI mixed F16/F32 auto/vendor dispatch rows vs direct rocBLAS reference (wrapper-semantics matrix)\n");
    std::printf("  CSR_SPMV_F32: adaptive generated SpMV (thread-per-row or subgroup-per-row) vs rocSPARSE rocsparse_v2_spmv\n");
    std::printf("  HIPSPARSELT_STRUCTURED_GEMM_F16: hipSPARSELt structured sparse matmul vs rocBLAS dense pruned-A reference (opt-in)\n");
    std::printf("  CSR_SPMV_F32_CFD_FIXTURE: persisted CSR pressure-matrix fixture vs rocSPARSE (fixture mode)\n");
    std::printf("benchmark matrices: standard, broad, cta128-evidence, edge-tails, wrapper-semantics, cfd-fixture (quick mode overrides matrix to smoke shapes)\n");
    std::printf("hipSPARSELt support: %s (mode: capability_reporting_only)\n",
                NAVATALA_HAVE_HIPSPARSELT ? "available" : "unavailable");
}

template <typename T>
T readPod(std::ifstream& in, const char* label) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        std::fprintf(stderr, "ERROR: failed to read CSR fixture field %s\n", label);
        std::exit(2);
    }
    return value;
}

template <typename T>
std::vector<T> readVector(std::ifstream& in, std::size_t count, const char* label) {
    std::vector<T> out(count);
    if (count == 0) {
        return out;
    }
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        std::fprintf(stderr, "ERROR: failed to read CSR fixture vector %s\n", label);
        std::exit(2);
    }
    return out;
}

std::string basenameOf(const std::string& path) {
    const std::size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

CsrFixture loadCsrFixture(const Options& options) {
    CsrFixture fixture;
    fixture.jsonPath = options.csrFixtureJsonPath;
    fixture.binPath = options.csrFixtureBinPath;

    std::ifstream in(options.csrFixtureBinPath, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "ERROR: could not open CSR fixture binary: %s\n", options.csrFixtureBinPath.c_str());
        std::exit(2);
    }

    std::array<char, 16> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 16> expectedMagic = {'N','V','C','S','R','S','P','M','V','F','3','2','\0','\0','\0','\0'};
    if (!in || magic != expectedMagic) {
        std::fprintf(stderr, "ERROR: invalid CSR fixture magic in %s\n", options.csrFixtureBinPath.c_str());
        std::exit(2);
    }

    const std::uint32_t version = readPod<std::uint32_t>(in, "version");
    fixture.indexBase = readPod<std::uint32_t>(in, "indexBase");
    (void)readPod<std::uint32_t>(in, "flags");
    (void)readPod<std::uint32_t>(in, "reserved");
    fixture.rows = readPod<std::uint64_t>(in, "rows");
    fixture.cols = readPod<std::uint64_t>(in, "cols");
    fixture.nnz = readPod<std::uint64_t>(in, "nnz");

    if (version != 1) {
        std::fprintf(stderr, "ERROR: unsupported CSR fixture version %u\n", version);
        std::exit(2);
    }
    if (fixture.indexBase != 0) {
        std::fprintf(stderr, "ERROR: only zero-based CSR fixtures are supported, got indexBase=%u\n", fixture.indexBase);
        std::exit(2);
    }
    if (fixture.rows == 0 || fixture.cols == 0) {
        std::fprintf(stderr, "ERROR: CSR fixture rows/cols must be non-zero\n");
        std::exit(2);
    }
    if (fixture.rows > std::numeric_limits<std::uint32_t>::max() ||
        fixture.cols > std::numeric_limits<std::uint32_t>::max() ||
        fixture.nnz > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr, "ERROR: CSR fixture exceeds uint32 kernel limits\n");
        std::exit(2);
    }

    fixture.rowPtr = readVector<std::uint32_t>(in, static_cast<std::size_t>(fixture.rows + 1), "rowPtr");
    fixture.colIdx = readVector<std::uint32_t>(in, static_cast<std::size_t>(fixture.nnz), "colIdx");
    fixture.values = readVector<float>(in, static_cast<std::size_t>(fixture.nnz), "values");
    fixture.x = readVector<float>(in, static_cast<std::size_t>(fixture.cols), "x");

    if (fixture.rowPtr.front() != 0 || fixture.rowPtr.back() != fixture.nnz) {
        std::fprintf(stderr, "ERROR: CSR fixture rowPtr endpoints are invalid\n");
        std::exit(2);
    }
    for (std::size_t r = 0; r + 1 < fixture.rowPtr.size(); ++r) {
        if (fixture.rowPtr[r] > fixture.rowPtr[r + 1]) {
            std::fprintf(stderr, "ERROR: CSR fixture rowPtr is not monotonic at row %zu\n", r);
            std::exit(2);
        }
    }
    for (std::uint32_t col : fixture.colIdx) {
        if (col >= fixture.cols) {
            std::fprintf(stderr, "ERROR: CSR fixture column index %u exceeds cols=%llu\n",
                         col,
                         static_cast<unsigned long long>(fixture.cols));
            std::exit(2);
        }
    }
    return fixture;
}

RowDistribution computeRowDistribution(
    const std::vector<std::uint32_t>& rowPtr,
    std::uint32_t subgroupThreshold) {
    RowDistribution dist;
    if (rowPtr.size() < 2) {
        return dist;
    }
    std::vector<std::uint32_t> rowNnz(rowPtr.size() - 1);
    std::uint64_t total = 0;
    for (std::size_t r = 0; r < rowNnz.size(); ++r) {
        rowNnz[r] = rowPtr[r + 1] - rowPtr[r];
        total += rowNnz[r];
        if (rowNnz[r] >= subgroupThreshold) {
            ++dist.subgroupRows;
        } else {
            ++dist.threadRows;
        }
    }
    std::sort(rowNnz.begin(), rowNnz.end());
    auto percentile = [&](double p) -> std::uint32_t {
        if (rowNnz.empty()) return 0;
        const double raw = p * static_cast<double>(rowNnz.size() - 1);
        const std::size_t idx = static_cast<std::size_t>(std::llround(raw));
        return rowNnz[std::min(idx, rowNnz.size() - 1)];
    };
    dist.minNnz = rowNnz.front();
    dist.maxNnz = rowNnz.back();
    dist.meanNnz = static_cast<double>(total) / static_cast<double>(rowNnz.size());
    for (std::size_t d = 0; d < dist.decileNnz.size(); ++d) {
        dist.decileNnz[d] = percentile(static_cast<double>(d) / 10.0);
    }
    dist.p50Nnz = percentile(0.50);
    dist.p90Nnz = percentile(0.90);
    dist.p99Nnz = percentile(0.99);
    return dist;
}

std::string rowDistributionNotes(
    const RowDistribution& dist,
    std::uint32_t subgroupThreshold,
    const std::string& jsonPath) {
    std::ostringstream os;
    os << "real CSR fixture";
    if (!jsonPath.empty()) {
        os << "; sidecar=" << basenameOf(jsonPath);
    }
    os << "; rowNnz[min/mean/p50/p90/p99/max]="
       << dist.minNnz << "/"
       << std::fixed << std::setprecision(2) << dist.meanNnz << "/"
       << dist.p50Nnz << "/"
       << dist.p90Nnz << "/"
       << dist.p99Nnz << "/"
       << dist.maxNnz
       << "; threshold=" << subgroupThreshold
       << "; decileRowNnz[0,10,...,100]=";
    for (std::size_t d = 0; d < dist.decileNnz.size(); ++d) {
        if (d != 0) os << ",";
        os << dist.decileNnz[d];
    }
    os << "; decilePath[0,10,...,100]=";
    for (std::size_t d = 0; d < dist.decileNnz.size(); ++d) {
        if (d != 0) os << ",";
        os << (dist.decileNnz[d] >= subgroupThreshold ? "subgroup" : "thread");
    }
    os
       << "; threshold_deciles/threadRows=" << dist.threadRows
       << "; subgroupRows=" << dist.subgroupRows;
    return os.str();
}

std::uint32_t parseU32Env(const char* name, std::uint32_t defaultValue) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return defaultValue;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0') {
        std::fprintf(stderr,
                     "WARN: ignoring invalid %s=%s; using %u\n",
                     name,
                     raw,
                     defaultValue);
        return defaultValue;
    }
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr,
                     "WARN: %s=%s exceeds uint32; using %u\n",
                     name,
                     raw,
                     defaultValue);
        return defaultValue;
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint32_t spmvSubgroupThreshold() {
    return parseU32Env("NAVATALA_GPU_SPMV_SUBGROUP_THRESHOLD", 16u);
}

class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        if (old) {
            hadOld_ = true;
            oldValue_ = old;
        }
        setenv(name, value, 1);
    }

    ~ScopedEnv() {
        if (hadOld_) {
            setenv(name_, oldValue_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool hadOld_ = false;
    std::string oldValue_;
};

template <typename F>
double measureMs(int warmup, int iterations, F&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    check("hipDeviceSynchronize(warmup)", hipDeviceSynchronize());

    hipEvent_t start{};
    hipEvent_t stop{};
    check("hipEventCreate(start)", hipEventCreate(&start));
    check("hipEventCreate(stop)", hipEventCreate(&stop));
    check("hipEventRecord(start)", hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    check("hipEventRecord(stop)", hipEventRecord(stop));
    check("hipEventSynchronize(stop)", hipEventSynchronize(stop));
    float elapsedMs = 0.0f;
    check("hipEventElapsedTime", hipEventElapsedTime(&elapsedMs, start, stop));
    check("hipEventDestroy(start)", hipEventDestroy(start));
    check("hipEventDestroy(stop)", hipEventDestroy(stop));
    return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

template <typename F>
double measureMsOnStream(hipStream_t stream, int warmup, int iterations, F&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    check("hipStreamSynchronize(warmup)", hipStreamSynchronize(stream));

    hipEvent_t start{};
    hipEvent_t stop{};
    check("hipEventCreate(start)", hipEventCreate(&start));
    check("hipEventCreate(stop)", hipEventCreate(&stop));
    check("hipEventRecord(start)", hipEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    check("hipEventRecord(stop)", hipEventRecord(stop, stream));
    check("hipEventSynchronize(stop)", hipEventSynchronize(stop));
    float elapsedMs = 0.0f;
    check("hipEventElapsedTime", hipEventElapsedTime(&elapsedMs, start, stop));
    check("hipEventDestroy(start)", hipEventDestroy(start));
    check("hipEventDestroy(stop)", hipEventDestroy(stop));
    return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

template <typename F>
StreamTiming measureMsOnStreamDetailed(hipStream_t stream, int warmup, int iterations, F&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    check("hipStreamSynchronize(warmup)", hipStreamSynchronize(stream));

    hipEvent_t start{};
    hipEvent_t stop{};
    check("hipEventCreate(start)", hipEventCreate(&start));
    check("hipEventCreate(stop)", hipEventCreate(&stop));
    check("hipEventRecord(start)", hipEventRecord(start, stream));
    const auto hostStart = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    check("hipEventRecord(stop)", hipEventRecord(stop, stream));
    const auto hostSubmitStop = std::chrono::steady_clock::now();
    check("hipEventSynchronize(stop)", hipEventSynchronize(stop));
    const auto hostWallStop = std::chrono::steady_clock::now();
    float elapsedMs = 0.0f;
    check("hipEventElapsedTime", hipEventElapsedTime(&elapsedMs, start, stop));
    check("hipEventDestroy(start)", hipEventDestroy(start));
    check("hipEventDestroy(stop)", hipEventDestroy(stop));

    const double denom = static_cast<double>(iterations);
    const auto submitNs = std::chrono::duration_cast<std::chrono::nanoseconds>(hostSubmitStop - hostStart).count();
    const auto wallNs = std::chrono::duration_cast<std::chrono::nanoseconds>(hostWallStop - hostStart).count();
    return StreamTiming{
        static_cast<double>(elapsedMs) / denom,
        static_cast<double>(submitNs) / 1.0e6 / denom,
        static_cast<double>(wallNs) / 1.0e6 / denom};
}

template <typename T>
T* deviceAlloc(std::size_t count) {
    T* ptr = nullptr;
    check("hipMalloc", hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

template <typename T>
void copyToDevice(T* dst, const std::vector<T>& src) {
    check("hipMemcpy H2D", hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copyToHost(const T* src, std::size_t count) {
    std::vector<T> out(count);
    check("hipMemcpy D2H", hipMemcpy(out.data(), src, count * sizeof(T), hipMemcpyDeviceToHost));
    return out;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return err;
}

const char* transposeShort(NavatalaMatrixTranspose op) {
    return op == NAVATALA_MATRIX_OP_TRANSPOSE ? "T" : "N";
}

rocblas_operation rocblasOpForRowMajorTranspose(NavatalaMatrixTranspose op) {
    return op == NAVATALA_MATRIX_OP_TRANSPOSE ? rocblas_operation_transpose : rocblas_operation_none;
}

rocblas_int storedColsForA(std::uint32_t m, std::uint32_t k, NavatalaMatrixTranspose op) {
    return static_cast<rocblas_int>(op == NAVATALA_MATRIX_OP_TRANSPOSE ? m : k);
}

rocblas_int storedColsForB(std::uint32_t n, std::uint32_t k, NavatalaMatrixTranspose op) {
    return static_cast<rocblas_int>(op == NAVATALA_MATRIX_OP_TRANSPOSE ? k : n);
}

std::uint16_t halfBits(const rocblas_half& value) {
#if defined(ROCM_USE_FLOAT16)
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#else
    return value.data;
#endif
}

float halfToFloat(const rocblas_half& value) {
    const std::uint16_t bits = halfBits(value);
    const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
    const std::uint16_t exp = static_cast<std::uint16_t>((bits >> 10) & 0x1fu);
    const std::uint16_t frac = static_cast<std::uint16_t>(bits & 0x03ffu);
    if (exp == 0) {
        return sign * std::ldexp(static_cast<float>(frac), -24);
    }
    if (exp == 31) {
        if (frac == 0) return sign * std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(frac) / 1024.0f, static_cast<int>(exp) - 15);
}

double maxAbsDiffHalf(const std::vector<rocblas_half>& a, const std::vector<rocblas_half>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(static_cast<double>(halfToFloat(a[i])) -
                                     static_cast<double>(halfToFloat(b[i]))));
    }
    return err;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream os;
    for (char c : value) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    return os.str();
}

std::string envString(const char* name) {
    if (const char* value = std::getenv(name)) {
        return value;
    }
    return "";
}

Result benchAxpy(const Options& options, rocblas_handle blas, std::uint32_t n) {
    constexpr float alpha = 1.75f;

    std::vector<float> hx(n);
    std::vector<float> hy(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        hx[i] = static_cast<float>(static_cast<int>(i % 251u) - 100) * 0.01f;
        hy[i] = static_cast<float>(static_cast<int>(i % 127u) - 60) * 0.02f;
    }

    float* dx = deviceAlloc<float>(n);
    float* dyGenerated = deviceAlloc<float>(n);
    float* dyVendor = deviceAlloc<float>(n);
    float* dalpha = deviceAlloc<float>(1);
    std::uint32_t* dn = deviceAlloc<std::uint32_t>(1);
    copyToDevice(dx, hx);
    copyToDevice(dyGenerated, hy);
    copyToDevice(dyVendor, hy);
    check("hipMemcpy alpha", hipMemcpy(dalpha, &alpha, sizeof(float), hipMemcpyHostToDevice));
    check("hipMemcpy n", hipMemcpy(dn, &n, sizeof(std::uint32_t), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_sparse_axpy_f32, grid, block, 0, 0, dx, dalpha, dn, dyGenerated);
    });
    check("generated AXPY launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        check("rocblas_saxpy", rocblas_saxpy(blas, static_cast<rocblas_int>(n), &alpha, dx, 1, dyVendor, 1));
    });

    auto generated = copyToHost(dyGenerated, n);
    auto vendor = copyToHost(dyVendor, n);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(dx);
    releaseHip(dyGenerated);
    releaseHip(dyVendor);
    releaseHip(dalpha);
    releaseHip(dn);

    return {"AXPY_F32",
            "n=" + std::to_string(n),
            "Navatala HIP kernel navatala_sparse_axpy_f32",
            "rocBLAS rocblas_saxpy",
            err <= 1.0e-5,
            generatedMs,
            vendorMs,
            vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
            err,
            ""};
}

Result benchGemm(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    std::vector<float> ha(static_cast<std::size_t>(m) * k);
    std::vector<float> hb(static_cast<std::size_t>(k) * n);
    std::vector<float> hc(static_cast<std::size_t>(m) * n, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) ha[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.05f;
    for (std::size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.04f;

    float* da = deviceAlloc<float>(ha.size());
    float* db = deviceAlloc<float>(hb.size());
    float* dcGenerated = deviceAlloc<float>(hc.size());
    float* dcVendor = deviceAlloc<float>(hc.size());
    std::uint32_t* dm = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dn = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dk = deviceAlloc<std::uint32_t>(1);
    float* dalpha = deviceAlloc<float>(1);
    float* dbeta = deviceAlloc<float>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    check("hipMemcpy m", hipMemcpy(dm, &m, sizeof(m), hipMemcpyHostToDevice));
    check("hipMemcpy n", hipMemcpy(dn, &n, sizeof(n), hipMemcpyHostToDevice));
    check("hipMemcpy k", hipMemcpy(dk, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy alpha", hipMemcpy(dalpha, &alpha, sizeof(alpha), hipMemcpyHostToDevice));
    check("hipMemcpy beta", hipMemcpy(dbeta, &beta, sizeof(beta), hipMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f32, grid, block, 0, 0,
                           da, db, dm, dn, dk, dalpha, dbeta, dcGenerated);
    });
    check("generated GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_sgemm",
              rocblas_sgemm(blas,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(n),
                            static_cast<rocblas_int>(m),
                            static_cast<rocblas_int>(k),
                            &alpha,
                            db,
                            static_cast<rocblas_int>(n),
                            da,
                            static_cast<rocblas_int>(k),
                            &beta,
                            dcVendor,
                            static_cast<rocblas_int>(n)));
    });

    auto generated = copyToHost(dcGenerated, hc.size());
    auto vendor = copyToHost(dcVendor, hc.size());
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dm);
    releaseHip(dn);
    releaseHip(dk);
    releaseHip(dalpha);
    releaseHip(dbeta);

    return {"GEMM_F32",
            "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k),
            "Navatala HIP kernel navatala_transformer_tiled_gemm_f32",
            "rocBLAS rocblas_sgemm",
            err <= 5.0e-3,
            generatedMs,
            vendorMs,
            vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
            err,
            ""};
}

Result benchGemmF16PortableF32Out(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dm = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dn = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dk = deviceAlloc<std::uint32_t>(1);
    float* dalpha = deviceAlloc<float>(1);
    float* dbeta = deviceAlloc<float>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    check("hipMemcpy F16 m", hipMemcpy(dm, &m, sizeof(m), hipMemcpyHostToDevice));
    check("hipMemcpy F16 n", hipMemcpy(dn, &n, sizeof(n), hipMemcpyHostToDevice));
    check("hipMemcpy F16 k", hipMemcpy(dk, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy F16 alpha", hipMemcpy(dalpha, &alpha, sizeof(alpha), hipMemcpyHostToDevice));
    check("hipMemcpy F16 beta", hipMemcpy(dbeta, &beta, sizeof(beta), hipMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_f32_out,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           dm,
                           dn,
                           dk,
                           dalpha,
                           dbeta,
                           dcGenerated);
    });
    check("generated F16/F32-output GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F32 output/F32 compute)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dm);
    releaseHip(dn);
    releaseHip(dk);
    releaseHip(dalpha);
    releaseHip(dbeta);

    Result result{"GEMM_F16_PORTABLE_F32OUT",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",output=F32,compute=F32",
                  "Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out",
                  "rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation",
                  err <= 1.0e-3,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "portable F16-input/F32-output tiled GEMM; denominator for MFMA speedup tracking"};
    result.kernelClass = "scalar";
    result.implementationKind = "portable_kernel";
    result.tuningPath = "portable_f16_f32out_tiled";
    return result;
}

#if NAVATALA_HAVE_MFMA_BENCHMARK
Result benchGemmF16Mfma(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 32u) != 0u || (n % 32u) != 0u || (k % 8u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA requires m,n multiples of 32 and k multiple of 8; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkTiles = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kTiles = k / 8u;
    check("hipMemcpy MFMA kTiles", hipMemcpy(dkTiles, &kTiles, sizeof(kTiles), hipMemcpyHostToDevice));
    check("hipMemcpy MFMA kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy MFMA nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(64);
    const dim3 grid(n / 32u, m / 32u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_k_loop,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           dkTiles,
                           dkStride,
                           dnStride,
                           dcGenerated);
    });
    check("generated MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkTiles);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 5.0e-1,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental full-tile HIP/gfx942 MFMA K-loop; no edge tiles, alpha/beta, transpose, or batching; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_32x32x8_f16_f32_k_loop";
    return result;
}

Result benchGemmF16MfmaCta64Direct(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 64u) != 0u || (n % 64u) != 0u || (k % 8u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA_CTA64_DIRECT requires m,n multiples of 64 and k multiple of 8; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkTiles = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kTiles = k / 8u;
    check("hipMemcpy CTA64 direct kTiles", hipMemcpy(dkTiles, &kTiles, sizeof(kTiles), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 direct kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 direct nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid(n / 64u, m / 64u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_direct,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           dkTiles,
                           dkStride,
                           dnStride,
                           dcGenerated);
    });
    check("generated CTA64 direct MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 CTA64 direct reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkTiles);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA_CTA64_DIRECT",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32,cta=64x64x8,direct_load=true",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 1.0e-3,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental HIP/gfx942 MFMA CTA64 direct-load kernel; four wave64s per CTA, one accumulator per wave, no shared staging; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct";
    return result;
}

Result benchGemmF16MfmaCta64Shared(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 64u) != 0u || (n % 64u) != 0u || (k % 8u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA_CTA64_SHARED requires m,n multiples of 64 and k multiple of 8; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkTiles = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kTiles = k / 8u;
    check("hipMemcpy CTA64 shared kTiles", hipMemcpy(dkTiles, &kTiles, sizeof(kTiles), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 shared kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 shared nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid(n / 64u, m / 64u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_shared,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           kTiles,
                           k,
                           n,
                           dcGenerated);
    });
    check("generated CTA64 shared MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 CTA64 shared reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkTiles);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA_CTA64_SHARED",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32,cta=64x64x8,lds_staged=true",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 1.0e-3,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental HIP/gfx942 MFMA CTA64 LDS-staged kernel; four wave64s per CTA, one accumulator per wave, 2 KB LDS staging; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared";
    return result;
}

Result benchGemmF16MfmaCta64Pipelined(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 64u) != 0u || (n % 64u) != 0u || (k % 8u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA_CTA64_PIPELINED requires m,n multiples of 64 and k multiple of 8; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkTiles = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kTiles = k / 8u;
    check("hipMemcpy CTA64 pipelined kTiles", hipMemcpy(dkTiles, &kTiles, sizeof(kTiles), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 pipelined kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 pipelined nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid(n / 64u, m / 64u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           dkTiles,
                           dkStride,
                           dnStride,
                           dcGenerated);
    });
    check("generated CTA64 pipelined MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 CTA64 pipelined reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkTiles);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA_CTA64_PIPELINED",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 1.0e-3,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental HIP/gfx942 MFMA CTA64 R6 staged-panel kernel; two-slot typed panels, b16 copy requests, exact dynamic copy-group matching; current generated HIP lowerer is synchronous until async/pipelined lowering is admitted; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined";
    return result;
}

Result benchGemmF16MfmaCta64SharedEarlyBarrier(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 64u) != 0u || (n % 64u) != 0u || (k % 8u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER requires m,n multiples of 64 and k multiple of 8; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkTiles = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kTiles = k / 8u;
    check("hipMemcpy CTA64 shared early-barrier kTiles", hipMemcpy(dkTiles, &kTiles, sizeof(kTiles), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 shared early-barrier kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 shared early-barrier nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid(n / 64u, m / 64u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           dkTiles,
                           dkStride,
                           dnStride,
                           dcGenerated);
    });
    check("generated CTA64 shared early-barrier MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 CTA64 shared early-barrier reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkTiles);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 1.0e-3,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental HIP/gfx942 MFMA CTA64 LDS-staged kernel with barrier moved after fragment loads; four wave64s per CTA, one accumulator per wave, 2 KB LDS staging; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier";
    return result;
}

Result benchGemmF16MfmaCta64SharedPadded(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 64u) != 0u || (n % 64u) != 0u || (k % 8u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA_CTA64_SHARED_PADDED requires m,n multiples of 64 and k multiple of 8; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkTiles = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kTiles = k / 8u;
    check("hipMemcpy CTA64 shared padded kTiles", hipMemcpy(dkTiles, &kTiles, sizeof(kTiles), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 shared padded kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy CTA64 shared padded nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid(n / 64u, m / 64u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           dkTiles,
                           dkStride,
                           dnStride,
                           dcGenerated);
    });
    check("generated CTA64 shared padded MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 CTA64 shared padded reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkTiles);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA_CTA64_SHARED_PADDED",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 1.0e-3,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental HIP/gfx942 MFMA CTA64 padded-LDS kernel; four wave64s per CTA, one accumulator per wave, 2192 bytes LDS staging; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded";
    return result;
}

Result benchGemmF16MfmaCta128(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    if ((m % 128u) != 0u || (n % 128u) != 0u || (k % 32u) != 0u) {
        std::fprintf(stderr,
                     "ERROR: GEMM_F16_MFMA_CTA128 requires m,n multiples of 128 and k multiple of 32; got m=%u,n=%u,k=%u\n",
                     m,
                     n,
                     k);
        std::exit(2);
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    std::uint32_t* dkBlocks = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dkStride = deviceAlloc<std::uint32_t>(1);
    std::uint32_t* dnStride = deviceAlloc<std::uint32_t>(1);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);
    const std::uint32_t kBlocks = k / 32u;
    check("hipMemcpy CTA128 kBlocks", hipMemcpy(dkBlocks, &kBlocks, sizeof(kBlocks), hipMemcpyHostToDevice));
    check("hipMemcpy CTA128 kStride", hipMemcpy(dkStride, &k, sizeof(k), hipMemcpyHostToDevice));
    check("hipMemcpy CTA128 nStride", hipMemcpy(dnStride, &n, sizeof(n), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid(n / 128u, m / 128u);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta128,
                           grid,
                           block,
                           0,
                           0,
                           da,
                           db,
                           kBlocks,
                           k,
                           n,
                           dcGenerated);
    });
    check("generated CTA128 MFMA GEMM launch", hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 CTA128 reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);
    releaseHip(dkBlocks);
    releaseHip(dkStride);
    releaseHip(dnStride);

    Result result{"GEMM_F16_MFMA_CTA128",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",compute=F32,cta=128x128x32",
                  "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 5.0e-1,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "experimental HIP/gfx942 MFMA CTA128 Phase-0 kernel; tile-divisible NN only, alpha=1, beta=0; benchmark is back-to-back throughput per launch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = "hip_mfma_gfx942_128x128x32_f16_f32_cta128";
    return result;
}

Result benchGemmF16MfmaEdge(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    bool cta128,
    bool cta64NnFast = false) {
    if (cta128 && cta64NnFast) {
        std::fprintf(stderr, "ERROR: CTA64 NN fast EDGE row cannot be requested with CTA128\n");
        std::exit(2);
    }
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr std::uint32_t transA = 0u;
    constexpr std::uint32_t transB = 0u;
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);

    const std::uint32_t kCount = cta128 ? ((k + 31u) / 32u) : ((k + 7u) / 8u);
    const std::uint32_t aStride = k;
    const std::uint32_t bStride = n;
    const std::uint32_t cStride = n;
    const std::uint32_t aBatchStride = static_cast<std::uint32_t>(aCount);
    const std::uint32_t bBatchStride = static_cast<std::uint32_t>(bCount);
    const std::uint32_t cBatchStride = static_cast<std::uint32_t>(cCount);
    const dim3 block(256);
    const dim3 grid(
        cta128 ? ((n + 127u) / 128u) : ((n + 63u) / 64u),
        cta128 ? ((m + 127u) / 128u) : ((m + 63u) / 64u),
        1);

    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        if (cta128) {
            hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta128_edge,
                               grid,
                               block,
                               0,
                               0,
                               da,
                               db,
                               m,
                               n,
                               k,
                               kCount,
                               aStride,
                               bStride,
                               cStride,
                               aBatchStride,
                               bBatchStride,
                               cBatchStride,
                               transA,
                               transB,
                               alpha,
                               beta,
                               dcGenerated);
        } else if (cta64NnFast) {
            hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn,
                               grid,
                               block,
                               0,
                               0,
                               da,
                               db,
                               m,
                               n,
                               k,
                               kCount,
                               aStride,
                               bStride,
                               cStride,
                               dcGenerated);
        } else {
            hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge,
                               grid,
                               block,
                               0,
                               0,
                               da,
                               db,
                               m,
                               n,
                               k,
                               kCount,
                               aStride,
                               bStride,
                               cStride,
                               aBatchStride,
                               bBatchStride,
                               cBatchStride,
                               transA,
                               transB,
                               alpha,
                               beta,
                               dcGenerated);
        }
    });
    check(cta128
              ? "generated CTA128 edge MFMA GEMM launch"
              : (cta64NnFast
                    ? "generated CTA64 NN fast edge MFMA GEMM launch"
                    : "generated CTA64 edge MFMA GEMM launch"),
          hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        check("rocblas_gemm_ex(F16/F32 edge reference)",
              rocblas_gemm_ex(blas,
                              rocblas_operation_none,
                              rocblas_operation_none,
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(n),
                              da,
                              rocblas_datatype_f16_r,
                              static_cast<rocblas_int>(k),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);

    const std::string operation = cta128
        ? "GEMM_F16_MFMA_CTA128_EDGE"
        : (cta64NnFast ? "GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN" : "GEMM_F16_MFMA_CTA64_SHARED_EDGE");
    const std::string generatedPath = cta128
        ? "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge"
        : (cta64NnFast
            ? "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn"
            : "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge");
    Result result{operation,
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) +
                      ",compute=F32,edge=true" + (cta64NnFast ? ",nnFast=true" : ""),
                  generatedPath,
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 5.0e-1,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  cta64NnFast
                      ? "NN-only CTA64 true-tail HIP/gfx942 MFMA candidate: alpha=1, beta=0, no transpose, no batching; direct-row evidence slot for EDGE instruction-count reduction"
                      : "edge-capable HIP/gfx942 MFMA kernel using by-value scalar ABI; direct-row baseline for public wrapper overhead"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = cta128
        ? "hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge"
        : (cta64NnFast
            ? "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn"
            : "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge");
    return result;
}

Result benchGemmF16MfmaSemantic(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    bool cta128,
    const std::string& semanticSuffix,
    float alpha,
    float beta,
    NavatalaMatrixTranspose transA,
    NavatalaMatrixTranspose transB) {
    const std::uint32_t tile = cta128 ? 128u : 64u;
    const std::uint32_t kStep = cta128 ? 32u : 8u;
    if ((m % tile) != 0u || (n % tile) != 0u || (k % kStep) != 0u) {
        std::fprintf(stderr,
                     "ERROR: %s semantic MFMA requires m,n multiples of %u and k multiple of %u; got m=%u,n=%u,k=%u\n",
                     cta128 ? "CTA128" : "CTA64",
                     tile,
                     kStep,
                     m,
                     n,
                     k);
        std::exit(2);
    }

    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<__half> ha(aCount);
    std::vector<__half> hb(bCount);
    std::vector<float> hc(cCount, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }
    for (std::size_t i = 0; i < hc.size(); ++i) {
        hc[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
    }

    __half* da = deviceAlloc<__half>(aCount);
    __half* db = deviceAlloc<__half>(bCount);
    float* dcGenerated = deviceAlloc<float>(cCount);
    float* dcVendor = deviceAlloc<float>(cCount);
    copyToDevice(da, ha);
    copyToDevice(db, hb);
    copyToDevice(dcGenerated, hc);
    copyToDevice(dcVendor, hc);

    const std::uint32_t kCount = k / kStep;
    const std::uint32_t aStride = static_cast<std::uint32_t>(storedColsForA(m, k, transA));
    const std::uint32_t bStride = static_cast<std::uint32_t>(storedColsForB(n, k, transB));
    const std::uint32_t cStride = n;
    const std::uint32_t transAFlag = transA == NAVATALA_MATRIX_OP_TRANSPOSE ? 1u : 0u;
    const std::uint32_t transBFlag = transB == NAVATALA_MATRIX_OP_TRANSPOSE ? 1u : 0u;
    const dim3 block(256);
    const dim3 grid(n / tile, m / tile);
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        if (cta128) {
            hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta128_semantic,
                               grid,
                               block,
                               0,
                               0,
                               da,
                               db,
                               m,
                               n,
                               kCount,
                               aStride,
                               bStride,
                               cStride,
                               transAFlag,
                               transBFlag,
                               alpha,
                               beta,
                               dcGenerated);
        } else {
            hipLaunchKernelGGL(navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_semantic,
                               grid,
                               block,
                               0,
                               0,
                               da,
                               db,
                               m,
                               n,
                               kCount,
                               aStride,
                               bStride,
                               cStride,
                               transAFlag,
                               transBFlag,
                               alpha,
                               beta,
                               dcGenerated);
        }
    });
    check(cta128 ? "generated CTA128 semantic MFMA GEMM launch" : "generated CTA64 semantic MFMA GEMM launch",
          hipGetLastError());

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        // Row-major C = op(A)*op(B) is column-major C^T = op(B)^T*op(A)^T.
        check("rocblas_gemm_ex(F16/F32 semantic reference)",
              rocblas_gemm_ex(blas,
                              rocblasOpForRowMajorTranspose(transB),
                              rocblasOpForRowMajorTranspose(transA),
                              static_cast<rocblas_int>(n),
                              static_cast<rocblas_int>(m),
                              static_cast<rocblas_int>(k),
                              &alpha,
                              db,
                              rocblas_datatype_f16_r,
                              storedColsForB(n, k, transB),
                              da,
                              rocblas_datatype_f16_r,
                              storedColsForA(m, k, transA),
                              &beta,
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              dcVendor,
                              rocblas_datatype_f32_r,
                              static_cast<rocblas_int>(n),
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard,
                              0,
                              0));
    });

    auto generated = copyToHost(dcGenerated, cCount);
    auto vendor = copyToHost(dcVendor, cCount);
    const double err = maxAbsDiff(generated, vendor);

    releaseHip(da);
    releaseHip(db);
    releaseHip(dcGenerated);
    releaseHip(dcVendor);

    const std::string ctaName = cta128 ? "CTA128" : "CTA64_SHARED";
    Result result{"GEMM_F16_MFMA_" + ctaName + "_SEMANTIC_" + semanticSuffix,
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) +
                      ",k=" + std::to_string(k) + ",transA=" +
                      transposeShort(transA) + ",transB=" + transposeShort(transB) +
                      ",alpha=" + std::to_string(alpha) + ",beta=" +
                      std::to_string(beta) + ",compute=F32,semantic_full_tile=true",
                  cta128
                      ? "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_semantic"
                      : "Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_semantic",
                  "rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation",
                  err <= 5.0e-1,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  "direct semantic full-tile HIP/gfx942 MFMA row; separates Phase 2 kernel cost from public wrapper dispatch"};
    result.kernelClass = "mfma_f16";
    result.implementationKind = "tuned_kernel";
    result.tuningPath = cta128
        ? "hip_mfma_gfx942_128x128x32_f16_f32_cta128_semantic"
        : "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_semantic";
    return result;
}
#endif

#if NAVATALA_HAVE_RUNTIME_WRAPPER
Result benchGemmWrapperVendor(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    std::vector<float> ha(static_cast<std::size_t>(m) * k);
    std::vector<float> hb(static_cast<std::size_t>(k) * n);
    std::vector<float> hc(static_cast<std::size_t>(m) * n, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) ha[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.05f;
    for (std::size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.04f;

    NavatalaGpuContext* ctx = nullptr;
    NavatalaGpuQueue* queue = nullptr;
    NavatalaGpuBuffer* da = nullptr;
    NavatalaGpuBuffer* db = nullptr;
    NavatalaGpuBuffer* dcWrapper = nullptr;

    check("navatala_gpu_create_context(HIP)", navatala_gpu_create_context(NAVATALA_BACKEND_HIP_FFI, 0, &ctx));
    check("navatala_gpu_create_queue", navatala_gpu_create_queue(ctx, -1, &queue));
    check("navatala_gpu_create_buffer(A)", navatala_gpu_create_buffer(ctx, ha.size() * sizeof(float), &da));
    check("navatala_gpu_create_buffer(B)", navatala_gpu_create_buffer(ctx, hb.size() * sizeof(float), &db));
    check("navatala_gpu_create_buffer(C)", navatala_gpu_create_buffer(ctx, hc.size() * sizeof(float), &dcWrapper));
    check("navatala_gpu_copy_h2d(A)", navatala_gpu_copy_h2d(da, ha.data(), ha.size() * sizeof(float), queue));
    check("navatala_gpu_copy_h2d(B)", navatala_gpu_copy_h2d(db, hb.data(), hb.size() * sizeof(float), queue));
    check("navatala_gpu_copy_h2d(C)", navatala_gpu_copy_h2d(dcWrapper, hc.data(), hc.size() * sizeof(float), queue));
    check("navatala_gpu_queue_sync(initial copies)", navatala_gpu_queue_sync(queue));

    uintptr_t streamHandle = 0;
    std::uint8_t streamSupported = 0;
    check("navatala_gpu_queue_native_handle", navatala_gpu_queue_native_handle(queue, &streamHandle, &streamSupported));
    if (!streamSupported || streamHandle == 0) {
        std::fprintf(stderr, "ERROR: Navatala HIP queue did not expose a native stream handle\n");
        std::exit(2);
    }
    hipStream_t stream = reinterpret_cast<hipStream_t>(streamHandle);

    {
        ScopedEnv mode("NAVATALA_GPU_GEMM_VENDOR_MODE", "vendor");
        ScopedEnv threshold("NAVATALA_GPU_GEMM_VENDOR_THRESHOLD_FLOPS", "0");
        const StreamTiming wrapperTiming = measureMsOnStreamDetailed(stream, options.warmup, options.iterations, [&]() {
            check("navatala_gpu_gemm_f32(vendor)",
                  navatala_gpu_gemm_f32(da, db, dcWrapper, m, n, k, alpha, beta, queue));
        });
        const double wrapperMs = wrapperTiming.eventMeanMs;
        check("navatala_gpu_queue_sync(wrapper GEMM)", navatala_gpu_queue_sync(queue));

        float* daVendor = deviceAlloc<float>(ha.size());
        float* dbVendor = deviceAlloc<float>(hb.size());
        float* dcVendor = deviceAlloc<float>(hc.size());
        copyToDevice(daVendor, ha);
        copyToDevice(dbVendor, hb);
        copyToDevice(dcVendor, hc);

        check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
        const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
            check("rocblas_sgemm(wrapper reference)",
                  rocblas_sgemm(blas,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                static_cast<rocblas_int>(n),
                                static_cast<rocblas_int>(m),
                                static_cast<rocblas_int>(k),
                                &alpha,
                                dbVendor,
                                static_cast<rocblas_int>(n),
                                daVendor,
                                static_cast<rocblas_int>(k),
                                &beta,
                                dcVendor,
                                static_cast<rocblas_int>(n)));
        });

        auto wrapper = std::vector<float>(hc.size());
        check("navatala_gpu_copy_d2h(C)", navatala_gpu_copy_d2h(dcWrapper, wrapper.data(), wrapper.size() * sizeof(float), queue));
        check("navatala_gpu_queue_sync(final copy)", navatala_gpu_queue_sync(queue));
        auto vendor = copyToHost(dcVendor, hc.size());
        const double err = maxAbsDiff(wrapper, vendor);

        releaseHip(daVendor);
        releaseHip(dbVendor);
        releaseHip(dcVendor);
        navatala_gpu_destroy_buffer(da);
        navatala_gpu_destroy_buffer(db);
        navatala_gpu_destroy_buffer(dcWrapper);
        navatala_gpu_destroy_queue(queue);
        navatala_gpu_destroy_context(ctx);

        Result result{"GEMM_F32_WRAPPER_VENDOR",
                      "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k),
                      "public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor",
                      "rocBLAS rocblas_sgemm",
                      err <= 5.0e-3,
                      wrapperMs,
                      vendorMs,
                      vendorMs > 0.0 ? wrapperMs / vendorMs : 0.0,
                      err,
                      "wrapper dispatch row; fails loudly if vendor BLAS dispatch is unavailable"};
        result.implementationKind = "vendor_library";
        result.kernelClass = "vendor_library";
        result.tuningPath = "vendor_dispatch";
        result.vendorDispatchSelected = true;
        result.hostSubmitMeanMs = wrapperTiming.hostSubmitMeanMs;
        result.hostWallMeanMs = wrapperTiming.hostWallMeanMs;
        return result;
    }
}

Result benchGemmF16F32WrapperMfma(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const std::string& operationSuffix = std::string(),
    float alpha = 1.0f,
    float beta = 0.0f,
    NavatalaMatrixTranspose transA = NAVATALA_MATRIX_OP_NONE,
    NavatalaMatrixTranspose transB = NAVATALA_MATRIX_OP_NONE,
    std::uint32_t batchCount = 1,
    const std::string& wrapperKind = "MFMA",
    const char* implValue = "mfma",
    bool vendorDispatchSelected = false,
    const char* mfmaModeValue = "auto") {
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;
    const std::size_t totalA = aCount * batchCount;
    const std::size_t totalB = bCount * batchCount;
    const std::size_t totalC = cCount * batchCount;

    std::vector<__half> ha(totalA);
    std::vector<__half> hb(totalB);
    std::vector<float> hc(totalC, 0.0f);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2half(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }
    for (std::size_t i = 0; i < hc.size(); ++i) {
        hc[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
    }

    NavatalaGpuContext* ctx = nullptr;
    NavatalaGpuQueue* queue = nullptr;
    NavatalaGpuBuffer* da = nullptr;
    NavatalaGpuBuffer* db = nullptr;
    NavatalaGpuBuffer* dcWrapper = nullptr;

    check("navatala_gpu_create_context(HIP)", navatala_gpu_create_context(NAVATALA_BACKEND_HIP_FFI, 0, &ctx));
    check("navatala_gpu_create_queue", navatala_gpu_create_queue(ctx, -1, &queue));
    check("navatala_gpu_create_buffer(F16 A)", navatala_gpu_create_buffer(ctx, ha.size() * sizeof(__half), &da));
    check("navatala_gpu_create_buffer(F16 B)", navatala_gpu_create_buffer(ctx, hb.size() * sizeof(__half), &db));
    check("navatala_gpu_create_buffer(F32 C)", navatala_gpu_create_buffer(ctx, hc.size() * sizeof(float), &dcWrapper));
    check("navatala_gpu_copy_h2d(F16 A)", navatala_gpu_copy_h2d(da, ha.data(), ha.size() * sizeof(__half), queue));
    check("navatala_gpu_copy_h2d(F16 B)", navatala_gpu_copy_h2d(db, hb.data(), hb.size() * sizeof(__half), queue));
    check("navatala_gpu_copy_h2d(F32 C)", navatala_gpu_copy_h2d(dcWrapper, hc.data(), hc.size() * sizeof(float), queue));
    check("navatala_gpu_queue_sync(initial F16/F32 copies)", navatala_gpu_queue_sync(queue));

    uintptr_t streamHandle = 0;
    std::uint8_t streamSupported = 0;
    check("navatala_gpu_queue_native_handle", navatala_gpu_queue_native_handle(queue, &streamHandle, &streamSupported));
    if (!streamSupported || streamHandle == 0) {
        std::fprintf(stderr, "ERROR: Navatala HIP queue did not expose a native stream handle\n");
        std::exit(2);
    }
    hipStream_t stream = reinterpret_cast<hipStream_t>(streamHandle);

    {
        ScopedEnv impl("NAVATALA_GPU_GEMM_IMPL", implValue);
        ScopedEnv vendorMode("NAVATALA_GPU_GEMM_VENDOR_MODE", "auto");
        ScopedEnv mfmaMode("NAVATALA_GPU_GEMM_MFMA_MODE", mfmaModeValue);
        const StreamTiming wrapperTiming = measureMsOnStreamDetailed(stream, options.warmup, options.iterations, [&]() {
            if (batchCount > 1) {
                check("navatala_gpu_gemm_f16_f32_strided_batched(mfma)",
                      navatala_gpu_gemm_f16_f32_strided_batched(
                          da, db, dcWrapper, m, n, k, alpha, beta,
                          transA, transB, aCount, bCount, cCount, batchCount, queue));
            } else if (transA != NAVATALA_MATRIX_OP_NONE || transB != NAVATALA_MATRIX_OP_NONE) {
                check("navatala_gpu_gemm_f16_f32_ex(mfma)",
                      navatala_gpu_gemm_f16_f32_ex(
                          da, db, dcWrapper, m, n, k, alpha, beta, transA, transB, queue));
            } else {
                check("navatala_gpu_gemm_f16_f32(mfma)",
                      navatala_gpu_gemm_f16_f32(da, db, dcWrapper, m, n, k, alpha, beta, queue));
            }
        });
        const double wrapperMs = wrapperTiming.eventMeanMs;
        check("navatala_gpu_queue_sync(F16/F32 wrapper GEMM)", navatala_gpu_queue_sync(queue));

        __half* daVendor = deviceAlloc<__half>(totalA);
        __half* dbVendor = deviceAlloc<__half>(totalB);
        float* dcVendor = deviceAlloc<float>(totalC);
        copyToDevice(daVendor, ha);
        copyToDevice(dbVendor, hb);
        copyToDevice(dcVendor, hc);

        check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
        const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
            // Row-major C = A*B is column-major C^T = B^T*A^T.
            if (batchCount > 1) {
                check("rocblas_gemm_strided_batched_ex(F16/F32 wrapper reference)",
                      rocblas_gemm_strided_batched_ex(
                          blas,
                          rocblasOpForRowMajorTranspose(transB),
                          rocblasOpForRowMajorTranspose(transA),
                          static_cast<rocblas_int>(n),
                          static_cast<rocblas_int>(m),
                          static_cast<rocblas_int>(k),
                          &alpha,
                          dbVendor,
                          rocblas_datatype_f16_r,
                          storedColsForB(n, k, transB),
                          static_cast<rocblas_stride>(bCount),
                          daVendor,
                          rocblas_datatype_f16_r,
                          storedColsForA(m, k, transA),
                          static_cast<rocblas_stride>(aCount),
                          &beta,
                          dcVendor,
                          rocblas_datatype_f32_r,
                          static_cast<rocblas_int>(n),
                          static_cast<rocblas_stride>(cCount),
                          dcVendor,
                          rocblas_datatype_f32_r,
                          static_cast<rocblas_int>(n),
                          static_cast<rocblas_stride>(cCount),
                          static_cast<rocblas_int>(batchCount),
                          rocblas_datatype_f32_r,
                          rocblas_gemm_algo_standard,
                          0,
                          0));
            } else {
                check("rocblas_gemm_ex(F16/F32 wrapper reference)",
                      rocblas_gemm_ex(blas,
                                      rocblasOpForRowMajorTranspose(transB),
                                      rocblasOpForRowMajorTranspose(transA),
                                      static_cast<rocblas_int>(n),
                                      static_cast<rocblas_int>(m),
                                      static_cast<rocblas_int>(k),
                                      &alpha,
                                      dbVendor,
                                      rocblas_datatype_f16_r,
                                      storedColsForB(n, k, transB),
                                      daVendor,
                                      rocblas_datatype_f16_r,
                                      storedColsForA(m, k, transA),
                                      &beta,
                                      dcVendor,
                                      rocblas_datatype_f32_r,
                                      static_cast<rocblas_int>(n),
                                      dcVendor,
                                      rocblas_datatype_f32_r,
                                      static_cast<rocblas_int>(n),
                                      rocblas_datatype_f32_r,
                                      rocblas_gemm_algo_standard,
                                      0,
                                      0));
            }
        });

        auto wrapper = std::vector<float>(hc.size());
        check("navatala_gpu_copy_d2h(F32 C)", navatala_gpu_copy_d2h(dcWrapper, wrapper.data(), wrapper.size() * sizeof(float), queue));
        check("navatala_gpu_queue_sync(final F16/F32 copy)", navatala_gpu_queue_sync(queue));
        auto vendor = copyToHost(dcVendor, hc.size());
        const double err = maxAbsDiff(wrapper, vendor);

        releaseHip(daVendor);
        releaseHip(dbVendor);
        releaseHip(dcVendor);
        navatala_gpu_destroy_buffer(da);
        navatala_gpu_destroy_buffer(db);
        navatala_gpu_destroy_buffer(dcWrapper);
        navatala_gpu_destroy_queue(queue);
        navatala_gpu_destroy_context(ctx);

        const std::string operation = operationSuffix.empty()
            ? "GEMM_F16_F32_WRAPPER_" + wrapperKind
            : "GEMM_F16_F32_WRAPPER_" + wrapperKind + "_" + operationSuffix;
        std::ostringstream shape;
        shape << "m=" << m
              << ",n=" << n
              << ",k=" << k
              << ",transA=" << transposeShort(transA)
              << ",transB=" << transposeShort(transB)
              << ",alpha=" << alpha
              << ",beta=" << beta
              << ",batch=" << batchCount
              << ",output=F32,compute=F32,wrapper=" << implValue;
        const std::string wrapperApi = batchCount > 1
            ? "navatala_gpu_gemm_f16_f32_strided_batched"
            : ((transA != NAVATALA_MATRIX_OP_NONE || transB != NAVATALA_MATRIX_OP_NONE)
                ? "navatala_gpu_gemm_f16_f32_ex"
                : "navatala_gpu_gemm_f16_f32");
        const bool splitDispatch = std::strcmp(mfmaModeValue, "split") == 0;

        Result result{operation,
                      shape.str(),
                      "public C ABI " + wrapperApi + " with NAVATALA_GPU_GEMM_IMPL=" +
                          std::string(implValue) +
                          ",NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=" +
                          std::string(mfmaModeValue),
                      batchCount > 1
                          ? "rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation"
                          : "rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation",
                      err <= 5.0e-1,
                      wrapperMs,
                      vendorMs,
                      vendorMs > 0.0 ? wrapperMs / vendorMs : 0.0,
                      err,
                      vendorDispatchSelected
                          ? "public wrapper vendor dispatch row via " + wrapperApi + "; validates mixed F16-input/F32-output rocBLAS routing through the C ABI"
                          : (splitDispatch
                              ? "public wrapper MFMA structural edge-split row via " + wrapperApi + "; uses CTA64 semantic full-K interior launches plus CTA64 edge-region launches for output borders and K tails; timing includes all split launches and wrapper scalar-parameter setup"
                              : "public wrapper MFMA dispatch row via " + wrapperApi + "; uses full-tile CTA64/CTA128 fast path for aligned alpha=1,beta=0 NN single-batch calls, full-tile semantic CTA64/CTA128 for tile-divisible alpha/beta or transpose single-batch calls, and edge-capable CTA64-first fallback otherwise; timing includes wrapper scalar-parameter setup and synchronization")};
        result.implementationKind = vendorDispatchSelected ? "vendor_library" : "tuned_kernel";
        result.kernelClass = vendorDispatchSelected ? "vendor_library" : "mfma_f16";
        result.tuningPath = vendorDispatchSelected
            ? "vendor_dispatch"
            : (splitDispatch
                ? "hip_mfma_gfx942_wrapper_dispatch_edge_split"
                : "hip_mfma_gfx942_wrapper_dispatch");
        result.vendorDispatchSelected = vendorDispatchSelected;
        result.hostSubmitMeanMs = wrapperTiming.hostSubmitMeanMs;
        result.hostWallMeanMs = wrapperTiming.hostWallMeanMs;
        return result;
    }
}
#endif

Result benchSpmv(
    const Options& options,
    const hipDeviceProp_t& props,
    rocsparse_handle sparse,
    std::uint32_t rows,
    std::uint32_t rowNnz) {
    const std::uint32_t cols = rows;
    const std::uint32_t nnz = rows * rowNnz;

    std::vector<std::uint32_t> hrow(rows + 1);
    std::vector<std::uint32_t> hcol(nnz);
    std::vector<float> hval(nnz);
    std::vector<float> hx(cols);
    for (std::uint32_t r = 0; r <= rows; ++r) hrow[r] = r * rowNnz;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t j = 0; j < rowNnz; ++j) {
            const std::uint32_t p = r * rowNnz + j;
            hcol[p] = (r + j * 17u) % cols;
            hval[p] = 0.1f + 0.01f * static_cast<float>(j);
        }
    }
    for (std::uint32_t i = 0; i < cols; ++i) hx[i] = static_cast<float>(static_cast<int>(i % 97u) - 48) * 0.01f;

    std::uint32_t* drow = deviceAlloc<std::uint32_t>(hrow.size());
    std::uint32_t* dcol = deviceAlloc<std::uint32_t>(hcol.size());
    float* dval = deviceAlloc<float>(hval.size());
    float* dx = deviceAlloc<float>(hx.size());
    float* dyGenerated = deviceAlloc<float>(rows);
    float* dyVendor = deviceAlloc<float>(rows);
    std::uint32_t* drows = deviceAlloc<std::uint32_t>(1);
    copyToDevice(drow, hrow);
    copyToDevice(dcol, hcol);
    copyToDevice(dval, hval);
    copyToDevice(dx, hx);
    check("hipMemset generated y", hipMemset(dyGenerated, 0, rows * sizeof(float)));
    check("hipMemset vendor y", hipMemset(dyVendor, 0, rows * sizeof(float)));
    check("hipMemcpy rows", hipMemcpy(drows, &rows, sizeof(rows), hipMemcpyHostToDevice));

    const std::uint32_t subgroupThreshold = spmvSubgroupThreshold();
    const bool useSubgroupPath = rowNnz >= subgroupThreshold;
    const std::uint32_t subgroupLaunchWidth =
        static_cast<std::uint32_t>(std::max(1, props.warpSize));
    const dim3 block(256);
    const dim3 grid(useSubgroupPath
                        ? ((rows * subgroupLaunchWidth + block.x - 1) / block.x)
                        : ((rows + block.x - 1) / block.x));
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        if (useSubgroupPath) {
            hipLaunchKernelGGL(navatala_graph_spmv_weighted_subgroup_f32, grid, block, 0, 0,
                               drow, dcol, dval, dx, drows, dyGenerated);
        } else {
            hipLaunchKernelGGL(navatala_graph_spmv_weighted_f32, grid, block, 0, 0,
                               drow, dcol, dval, dx, drows, dyGenerated);
        }
    });
    check("generated SpMV launch", hipGetLastError());

    rocsparse_spmat_descr mat = nullptr;
    rocsparse_dnvec_descr xvec = nullptr;
    rocsparse_dnvec_descr yvec = nullptr;
    rocsparse_spmv_descr spmvDescr = nullptr;
    check("rocsparse_create_csr_descr",
          rocsparse_create_csr_descr(&mat,
                                     rows,
                                     cols,
                                     nnz,
                                     drow,
                                     dcol,
                                     dval,
                                     rocsparse_indextype_i32,
                                     rocsparse_indextype_i32,
                                     rocsparse_index_base_zero,
                                     rocsparse_datatype_f32_r));
    check("rocsparse_create_dnvec_descr(x)",
          rocsparse_create_dnvec_descr(&xvec, cols, dx, rocsparse_datatype_f32_r));
    check("rocsparse_create_dnvec_descr(y)",
          rocsparse_create_dnvec_descr(&yvec, rows, dyVendor, rocsparse_datatype_f32_r));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    rocsparse_operation spmvOperation = rocsparse_operation_none;
    rocsparse_datatype spmvDatatype = rocsparse_datatype_f32_r;
    rocsparse_spmv_alg spmvAlg = rocsparse_spmv_alg_default;
    check("rocsparse_create_spmv_descr",
          rocsparse_create_spmv_descr(&spmvDescr));
    check("rocsparse_spmv_set_input operation",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_operation,
                                   &spmvOperation,
                                   sizeof(spmvOperation),
                                   nullptr));
    check("rocsparse_spmv_set_input scalar datatype",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_scalar_datatype,
                                   &spmvDatatype,
                                   sizeof(spmvDatatype),
                                   nullptr));
    check("rocsparse_spmv_set_input compute datatype",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_compute_datatype,
                                   &spmvDatatype,
                                   sizeof(spmvDatatype),
                                   nullptr));
    check("rocsparse_spmv_set_input algorithm",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_alg,
                                   &spmvAlg,
                                   sizeof(spmvAlg),
                                   nullptr));
    std::size_t analysisBytes = 0;
    std::size_t computeBytes = 0;
    check("rocsparse_v2_spmv analysis buffer size",
          rocsparse_v2_spmv_buffer_size(sparse,
                                        spmvDescr,
                                        mat,
                                        xvec,
                                        yvec,
                                        rocsparse_v2_spmv_stage_analysis,
                                        &analysisBytes,
                                        nullptr));
    check("rocsparse_v2_spmv compute buffer size",
          rocsparse_v2_spmv_buffer_size(sparse,
                                        spmvDescr,
                                        mat,
                                        xvec,
                                        yvec,
                                        rocsparse_v2_spmv_stage_compute,
                                        &computeBytes,
                                        nullptr));
    const std::size_t tempBytes = std::max(analysisBytes, computeBytes);
    void* temp = nullptr;
    if (tempBytes != 0) {
        check("hipMalloc SpMV temp", hipMalloc(&temp, tempBytes));
    }
    check("rocsparse_v2_spmv analysis",
          rocsparse_v2_spmv(sparse,
                            spmvDescr,
                            &alpha,
                            mat,
                            xvec,
                            &beta,
                            yvec,
                            rocsparse_v2_spmv_stage_analysis,
                            tempBytes,
                            temp,
                            nullptr));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        check("rocsparse_v2_spmv",
              rocsparse_v2_spmv(sparse,
                                spmvDescr,
                                &alpha,
                                mat,
                                xvec,
                                &beta,
                                yvec,
                                rocsparse_v2_spmv_stage_compute,
                                tempBytes,
                                temp,
                                nullptr));
    });

    auto generated = copyToHost(dyGenerated, rows);
    auto vendor = copyToHost(dyVendor, rows);
    const double err = maxAbsDiff(generated, vendor);

    if (temp) releaseHip(temp);
    rocsparse_destroy_spmv_descr(spmvDescr);
    rocsparse_destroy_dnvec_descr(xvec);
    rocsparse_destroy_dnvec_descr(yvec);
    rocsparse_destroy_spmat_descr(mat);
    releaseHip(drow);
    releaseHip(dcol);
    releaseHip(dval);
    releaseHip(dx);
    releaseHip(dyGenerated);
    releaseHip(dyVendor);
    releaseHip(drows);

    Result result{"CSR_SPMV_F32",
                  "rows=" + std::to_string(rows) + ",rowNnz=" + std::to_string(rowNnz) + ",nnz=" + std::to_string(nnz),
                  useSubgroupPath
                      ? "Navatala HIP kernel navatala_graph_spmv_weighted_subgroup_f32"
                      : "Navatala HIP kernel navatala_graph_spmv_weighted_f32",
                  "rocSPARSE rocsparse_v2_spmv",
                  err <= 1.0e-4,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  ""};
    result.tuningPath = useSubgroupPath ? "subgroup_per_row" : "thread_per_row";
    result.spmvRowNnzThreshold = subgroupThreshold;
    result.notes = "adaptive SpMV dispatch; override threshold with NAVATALA_GPU_SPMV_SUBGROUP_THRESHOLD";
    return result;
}

Result benchCsrFixtureSpmv(
    const Options& options,
    const hipDeviceProp_t& props,
    rocsparse_handle sparse,
    const CsrFixture& fixture,
    bool useSubgroupPath) {
    const std::uint32_t rows = static_cast<std::uint32_t>(fixture.rows);
    const std::uint32_t cols = static_cast<std::uint32_t>(fixture.cols);
    const std::uint32_t nnz = static_cast<std::uint32_t>(fixture.nnz);

    std::uint32_t* drow = deviceAlloc<std::uint32_t>(fixture.rowPtr.size());
    std::uint32_t* dcol = deviceAlloc<std::uint32_t>(fixture.colIdx.size());
    float* dval = deviceAlloc<float>(fixture.values.size());
    float* dx = deviceAlloc<float>(fixture.x.size());
    float* dyGenerated = deviceAlloc<float>(rows);
    float* dyVendor = deviceAlloc<float>(rows);
    std::uint32_t* drows = deviceAlloc<std::uint32_t>(1);
    copyToDevice(drow, fixture.rowPtr);
    copyToDevice(dcol, fixture.colIdx);
    copyToDevice(dval, fixture.values);
    copyToDevice(dx, fixture.x);
    check("hipMemset fixture generated y", hipMemset(dyGenerated, 0, rows * sizeof(float)));
    check("hipMemset fixture vendor y", hipMemset(dyVendor, 0, rows * sizeof(float)));
    check("hipMemcpy fixture rows", hipMemcpy(drows, &rows, sizeof(rows), hipMemcpyHostToDevice));

    const std::uint32_t subgroupThreshold = spmvSubgroupThreshold();
    const RowDistribution rowDist = computeRowDistribution(fixture.rowPtr, subgroupThreshold);
    const std::uint32_t subgroupLaunchWidth =
        static_cast<std::uint32_t>(std::max(1, props.warpSize));
    const dim3 block(256);
    const dim3 grid(useSubgroupPath
                        ? ((rows * subgroupLaunchWidth + block.x - 1) / block.x)
                        : ((rows + block.x - 1) / block.x));
    const double generatedMs = measureMs(options.warmup, options.iterations, [&]() {
        if (useSubgroupPath) {
            hipLaunchKernelGGL(navatala_graph_spmv_weighted_subgroup_f32, grid, block, 0, 0,
                               drow, dcol, dval, dx, drows, dyGenerated);
        } else {
            hipLaunchKernelGGL(navatala_graph_spmv_weighted_f32, grid, block, 0, 0,
                               drow, dcol, dval, dx, drows, dyGenerated);
        }
    });
    check("generated fixture SpMV launch", hipGetLastError());

    rocsparse_spmat_descr mat = nullptr;
    rocsparse_dnvec_descr xvec = nullptr;
    rocsparse_dnvec_descr yvec = nullptr;
    rocsparse_spmv_descr spmvDescr = nullptr;
    check("rocsparse_create_csr_descr(fixture)",
          rocsparse_create_csr_descr(&mat,
                                     rows,
                                     cols,
                                     nnz,
                                     drow,
                                     dcol,
                                     dval,
                                     rocsparse_indextype_i32,
                                     rocsparse_indextype_i32,
                                     rocsparse_index_base_zero,
                                     rocsparse_datatype_f32_r));
    check("rocsparse_create_dnvec_descr(fixture x)",
          rocsparse_create_dnvec_descr(&xvec, cols, dx, rocsparse_datatype_f32_r));
    check("rocsparse_create_dnvec_descr(fixture y)",
          rocsparse_create_dnvec_descr(&yvec, rows, dyVendor, rocsparse_datatype_f32_r));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    rocsparse_operation spmvOperation = rocsparse_operation_none;
    rocsparse_datatype spmvDatatype = rocsparse_datatype_f32_r;
    rocsparse_spmv_alg spmvAlg = rocsparse_spmv_alg_default;
    check("rocsparse_create_spmv_descr(fixture)",
          rocsparse_create_spmv_descr(&spmvDescr));
    check("rocsparse_spmv_set_input fixture operation",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_operation,
                                   &spmvOperation,
                                   sizeof(spmvOperation),
                                   nullptr));
    check("rocsparse_spmv_set_input fixture scalar datatype",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_scalar_datatype,
                                   &spmvDatatype,
                                   sizeof(spmvDatatype),
                                   nullptr));
    check("rocsparse_spmv_set_input fixture compute datatype",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_compute_datatype,
                                   &spmvDatatype,
                                   sizeof(spmvDatatype),
                                   nullptr));
    check("rocsparse_spmv_set_input fixture algorithm",
          rocsparse_spmv_set_input(sparse,
                                   spmvDescr,
                                   rocsparse_spmv_input_alg,
                                   &spmvAlg,
                                   sizeof(spmvAlg),
                                   nullptr));
    std::size_t analysisBytes = 0;
    std::size_t computeBytes = 0;
    check("rocsparse_v2_spmv fixture analysis buffer size",
          rocsparse_v2_spmv_buffer_size(sparse,
                                        spmvDescr,
                                        mat,
                                        xvec,
                                        yvec,
                                        rocsparse_v2_spmv_stage_analysis,
                                        &analysisBytes,
                                        nullptr));
    check("rocsparse_v2_spmv fixture compute buffer size",
          rocsparse_v2_spmv_buffer_size(sparse,
                                        spmvDescr,
                                        mat,
                                        xvec,
                                        yvec,
                                        rocsparse_v2_spmv_stage_compute,
                                        &computeBytes,
                                        nullptr));
    const std::size_t tempBytes = std::max(analysisBytes, computeBytes);
    void* temp = nullptr;
    if (tempBytes != 0) {
        check("hipMalloc fixture SpMV temp", hipMalloc(&temp, tempBytes));
    }
    check("rocsparse_v2_spmv fixture analysis",
          rocsparse_v2_spmv(sparse,
                            spmvDescr,
                            &alpha,
                            mat,
                            xvec,
                            &beta,
                            yvec,
                            rocsparse_v2_spmv_stage_analysis,
                            tempBytes,
                            temp,
                            nullptr));
    const double vendorMs = measureMs(options.warmup, options.iterations, [&]() {
        check("rocsparse_v2_spmv fixture",
              rocsparse_v2_spmv(sparse,
                                spmvDescr,
                                &alpha,
                                mat,
                                xvec,
                                &beta,
                                yvec,
                                rocsparse_v2_spmv_stage_compute,
                                tempBytes,
                                temp,
                                nullptr));
    });

    auto generated = copyToHost(dyGenerated, rows);
    auto vendor = copyToHost(dyVendor, rows);
    const double err = maxAbsDiff(generated, vendor);

    if (temp) releaseHip(temp);
    rocsparse_destroy_spmv_descr(spmvDescr);
    rocsparse_destroy_dnvec_descr(xvec);
    rocsparse_destroy_dnvec_descr(yvec);
    rocsparse_destroy_spmat_descr(mat);
    releaseHip(drow);
    releaseHip(dcol);
    releaseHip(dval);
    releaseHip(dx);
    releaseHip(dyGenerated);
    releaseHip(dyVendor);
    releaseHip(drows);

    Result result{"CSR_SPMV_F32_CFD_FIXTURE",
                  "path=" + std::string(useSubgroupPath ? "subgroup_per_row" : "thread_per_row") +
                      ",fixture=" + basenameOf(fixture.binPath) + ",rows=" + std::to_string(rows) +
                      ",cols=" + std::to_string(cols) + ",nnz=" + std::to_string(nnz),
                  useSubgroupPath
                      ? "Navatala HIP kernel navatala_graph_spmv_weighted_subgroup_f32"
                      : "Navatala HIP kernel navatala_graph_spmv_weighted_f32",
                  "rocSPARSE rocsparse_v2_spmv",
                  err <= 1.0e-4,
                  generatedMs,
                  vendorMs,
                  vendorMs > 0.0 ? generatedMs / vendorMs : 0.0,
                  err,
                  rowDistributionNotes(rowDist, subgroupThreshold, fixture.jsonPath)};
    result.tuningPath = useSubgroupPath ? "subgroup_per_row" : "thread_per_row";
    result.spmvRowNnzThreshold = subgroupThreshold;
    return result;
}

#if NAVATALA_HAVE_HIPSPARSELT
Result benchHipSparseLtStructuredGemmF16(
    const Options& options,
    rocblas_handle blas,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    constexpr float sparseLtAlpha = 1.0f;
    constexpr float sparseLtBeta = 0.0f;
    const rocblas_half blasAlpha = static_cast<rocblas_half>(1.0f);
    const rocblas_half blasBeta = static_cast<rocblas_half>(0.0f);
    const std::size_t aCount = static_cast<std::size_t>(m) * k;
    const std::size_t bCount = static_cast<std::size_t>(k) * n;
    const std::size_t cCount = static_cast<std::size_t>(m) * n;

    std::vector<rocblas_half> ha(aCount);
    std::vector<rocblas_half> hb(bCount);
    std::vector<rocblas_half> hc(cCount, static_cast<rocblas_half>(0.0f));
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = static_cast<rocblas_half>(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.025f);
    }
    for (std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = static_cast<rocblas_half>(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.02f);
    }

    rocblas_half* daDense = deviceAlloc<rocblas_half>(aCount);
    rocblas_half* daPruned = deviceAlloc<rocblas_half>(aCount);
    rocblas_half* db = deviceAlloc<rocblas_half>(bCount);
    rocblas_half* dcReference = deviceAlloc<rocblas_half>(cCount);
    rocblas_half* dcSparseLtInput = deviceAlloc<rocblas_half>(cCount);
    rocblas_half* dcSparseLtOutput = deviceAlloc<rocblas_half>(cCount);
    copyToDevice(daDense, ha);
    copyToDevice(db, hb);
    copyToDevice(dcReference, hc);
    copyToDevice(dcSparseLtInput, hc);
    copyToDevice(dcSparseLtOutput, hc);

    hipsparseLtHandle_t sparseLt{};
    checkHipSparseLt("hipsparseLtInit", hipsparseLtInit(&sparseLt));

    hipsparseLtMatDescriptor_t matA{};
    hipsparseLtMatDescriptor_t matB{};
    hipsparseLtMatDescriptor_t matC{};
    hipsparseLtMatDescriptor_t matD{};
    checkHipSparseLt("hipsparseLtStructuredDescriptorInit(A)",
                     hipsparseLtStructuredDescriptorInit(&sparseLt,
                                                         &matA,
                                                         m,
                                                         k,
                                                         k,
                                                         16,
                                                         HIP_R_16F,
                                                         HIPSPARSE_ORDER_ROW,
                                                         HIPSPARSELT_SPARSITY_50_PERCENT));
    checkHipSparseLt("hipsparseLtDenseDescriptorInit(B)",
                     hipsparseLtDenseDescriptorInit(&sparseLt,
                                                    &matB,
                                                    k,
                                                    n,
                                                    n,
                                                    16,
                                                    HIP_R_16F,
                                                    HIPSPARSE_ORDER_ROW));
    checkHipSparseLt("hipsparseLtDenseDescriptorInit(C)",
                     hipsparseLtDenseDescriptorInit(&sparseLt,
                                                    &matC,
                                                    m,
                                                    n,
                                                    n,
                                                    16,
                                                    HIP_R_16F,
                                                    HIPSPARSE_ORDER_ROW));
    checkHipSparseLt("hipsparseLtDenseDescriptorInit(D)",
                     hipsparseLtDenseDescriptorInit(&sparseLt,
                                                    &matD,
                                                    m,
                                                    n,
                                                    n,
                                                    16,
                                                    HIP_R_16F,
                                                    HIPSPARSE_ORDER_ROW));

    hipsparseLtMatmulDescriptor_t matmul{};
    checkHipSparseLt("hipsparseLtMatmulDescriptorInit",
                     hipsparseLtMatmulDescriptorInit(&sparseLt,
                                                     &matmul,
                                                     HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                     HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &matA,
                                                     &matB,
                                                     &matC,
                                                     &matD,
                                                     HIPSPARSELT_COMPUTE_32F));
    hipsparseLtMatmulAlgSelection_t alg{};
    checkHipSparseLt("hipsparseLtMatmulAlgSelectionInit",
                     hipsparseLtMatmulAlgSelectionInit(&sparseLt,
                                                       &alg,
                                                       &matmul,
                                                       HIPSPARSELT_MATMUL_ALG_DEFAULT));
    hipsparseLtMatmulPlan_t plan{};
    checkHipSparseLt("hipsparseLtMatmulPlanInit",
                     hipsparseLtMatmulPlanInit(&sparseLt, &plan, &matmul, &alg));

    hipStream_t stream = nullptr;
    checkHipSparseLt("hipsparseLtSpMMAPrune",
                     hipsparseLtSpMMAPrune(&sparseLt,
                                           &matmul,
                                           daDense,
                                           daPruned,
                                           HIPSPARSELT_PRUNE_SPMMA_STRIP,
                                           stream));
    check("hipDeviceSynchronize(prune)", hipDeviceSynchronize());

    int* dValid = deviceAlloc<int>(1);
    checkHipSparseLt("hipsparseLtSpMMAPruneCheck",
                     hipsparseLtSpMMAPruneCheck(&sparseLt, &matmul, daPruned, dValid, stream));
    int valid = 1;
    check("hipMemcpy prune valid", hipMemcpy(&valid, dValid, sizeof(int), hipMemcpyDeviceToHost));
    check("hipDeviceSynchronize(prune check)", hipDeviceSynchronize());
    if (valid != 0) {
        std::fprintf(stderr, "ERROR: hipSPARSELt prune check failed for m=%u,n=%u,k=%u\n", m, n, k);
        std::exit(2);
    }

    size_t compressedBytes = 0;
    size_t compressBufferBytes = 0;
    checkHipSparseLt("hipsparseLtSpMMACompressedSize",
                     hipsparseLtSpMMACompressedSize(&sparseLt, &plan, &compressedBytes, &compressBufferBytes));
    void* daCompressed = nullptr;
    void* compressBuffer = nullptr;
    check("hipMalloc compressed A", hipMalloc(&daCompressed, compressedBytes));
    if (compressBufferBytes != 0) {
        check("hipMalloc compress buffer", hipMalloc(&compressBuffer, compressBufferBytes));
    }
    checkHipSparseLt("hipsparseLtSpMMACompress",
                     hipsparseLtSpMMACompress(&sparseLt, &plan, daPruned, daCompressed, compressBuffer, stream));
    check("hipDeviceSynchronize(compress)", hipDeviceSynchronize());

    size_t workspaceBytes = 0;
    checkHipSparseLt("hipsparseLtMatmulGetWorkspace",
                     hipsparseLtMatmulGetWorkspace(&sparseLt, &plan, &workspaceBytes));
    void* workspace = nullptr;
    if (workspaceBytes != 0) {
        check("hipMalloc hipSPARSELt workspace", hipMalloc(&workspace, workspaceBytes));
    }

    hipStream_t streams[] = {stream};
    const double sparseLtMs = measureMs(options.warmup, options.iterations, [&]() {
        checkHipSparseLt("hipsparseLtMatmul",
                         hipsparseLtMatmul(&sparseLt,
                                           &plan,
                                           &sparseLtAlpha,
                                           daCompressed,
                                           db,
                                           &sparseLtBeta,
                                           dcSparseLtInput,
                                           dcSparseLtOutput,
                                           workspace,
                                           streams,
                                           1));
    });

    check("rocBLAS pointer mode host", rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    const double denseMs = measureMs(options.warmup, options.iterations, [&]() {
        check("rocblas_hgemm(pruned A reference)",
              rocblas_hgemm(blas,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(n),
                            static_cast<rocblas_int>(m),
                            static_cast<rocblas_int>(k),
                            &blasAlpha,
                            db,
                            static_cast<rocblas_int>(n),
                            daPruned,
                            static_cast<rocblas_int>(k),
                            &blasBeta,
                            dcReference,
                            static_cast<rocblas_int>(n)));
    });

    auto sparseLtOut = copyToHost(dcSparseLtOutput, cCount);
    auto dense = copyToHost(dcReference, cCount);
    const double err = maxAbsDiffHalf(sparseLtOut, dense);

    if (workspace) releaseHip(workspace);
    if (compressBuffer) releaseHip(compressBuffer);
    releaseHip(daCompressed);
    releaseHip(dValid);
    hipsparseLtMatmulPlanDestroy(&plan);
    hipsparseLtMatDescriptorDestroy(&matD);
    hipsparseLtMatDescriptorDestroy(&matC);
    hipsparseLtMatDescriptorDestroy(&matB);
    hipsparseLtMatDescriptorDestroy(&matA);
    hipsparseLtDestroy(&sparseLt);
    releaseHip(daDense);
    releaseHip(daPruned);
    releaseHip(db);
    releaseHip(dcReference);
    releaseHip(dcSparseLtInput);
    releaseHip(dcSparseLtOutput);

    Result result{"HIPSPARSELT_STRUCTURED_GEMM_F16",
                  "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",sparsity=50%,compute=F32",
                  "hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul",
                  "rocBLAS rocblas_hgemm dense pruned-A reference",
                  err <= 5.0e-2,
                  sparseLtMs,
                  denseMs,
                  denseMs > 0.0 ? sparseLtMs / denseMs : 0.0,
                  err,
                  "hipSPARSELt benchmark row; setup/prune/compress are outside timed matmul loop"};
    result.implementationKind = "vendor_library";
    result.kernelClass = "vendor_library";
    result.tuningPath = "vendor_dispatch";
    result.vendorDispatchSelected = true;
    return result;
}
#endif

void printResults(const std::vector<Result>& results) {
    std::printf("%-14s %-24s %-12s %-12s %-12s %-10s %s\n",
                "operation",
                "shape",
                "generated_mean_ms",
                "vendor_mean_ms",
                "ratio",
                "correct",
                "vendor");
    for (const auto& r : results) {
        std::printf("%-14s %-24s %-12.6f %-12.6f %-12.3f %-10s %s\n",
                    r.operation.c_str(),
                    r.shape.c_str(),
                    r.navatalaMs,
                    r.vendorMs,
                    r.generatedOverVendorRatio,
                    r.correctness ? "pass" : "fail",
                    r.vendorPath.c_str());
    }
}

void writeJson(
    const std::string& path,
    const Options& options,
    const hipDeviceProp_t& props,
    const std::vector<Result>& results) {
    if (path.empty()) return;
    std::ofstream out(path);
    out << "{\n";
    out << "  \"schema\": \"navatala_gpu.rocm_vendor_benchmark.v1\",\n";
    out << "  \"timingMode\": \"back_to_back_throughput_mean_per_launch\",\n";
    out << "  \"iterations\": " << options.iterations << ",\n";
    out << "  \"warmup\": " << options.warmup << ",\n";
    out << "  \"quickMode\": " << (options.quick ? "true" : "false") << ",\n";
    out << "  \"matrix\": \"" << jsonEscape(options.quick ? "quick" : options.matrix) << "\",\n";
    out << "  \"device\": {\n";
    out << "    \"name\": \"" << jsonEscape(props.name) << "\",\n";
    out << "    \"gcnArch\": \"" << jsonEscape(props.gcnArchName) << "\",\n";
    out << "    \"globalMemoryMiB\": " << static_cast<std::size_t>(props.totalGlobalMem / (1024 * 1024)) << "\n";
    out << "  },\n";
    int hipRuntimeVersion = 0;
    int hipDriverVersion = 0;
    (void)hipRuntimeGetVersion(&hipRuntimeVersion);
    (void)hipDriverGetVersion(&hipDriverVersion);
    out << "  \"rocminfo\": {\n";
    out << "    \"gcnArch\": \"" << jsonEscape(props.gcnArchName) << "\",\n";
    out << "    \"hipRuntimeVersion\": " << hipRuntimeVersion << ",\n";
    out << "    \"hipDriverVersion\": " << hipDriverVersion << ",\n";
    out << "    \"summary\": \"" << jsonEscape(envString("NAVATALA_GPU_ROCMINFO_SUMMARY")) << "\"\n";
    out << "  },\n";
    out << "  \"hipSPARSELtAvailable\": " << (NAVATALA_HAVE_HIPSPARSELT ? "true" : "false") << ",\n";
    out << "  \"hipSPARSELtMode\": \""
        << (options.includeHipSparseLtBenchmark && NAVATALA_HAVE_HIPSPARSELT ? "vendor_benchmark" : "capability_reporting_only")
        << "\",\n";
    out << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"operation\": \"" << jsonEscape(r.operation) << "\",\n";
        out << "      \"shape\": \"" << jsonEscape(r.shape) << "\",\n";
        out << "      \"navatalaPath\": \"" << jsonEscape(r.navatalaPath) << "\",\n";
        out << "      \"vendorPath\": \"" << jsonEscape(r.vendorPath) << "\",\n";
        out << "      \"correctness\": " << (r.correctness ? "true" : "false") << ",\n";
        out << "      \"generatedMeanMs\": " << std::setprecision(8) << r.navatalaMs << ",\n";
        out << "      \"vendorMeanMs\": " << std::setprecision(8) << r.vendorMs << ",\n";
        out << "      \"generatedOverVendorRatio\": " << std::setprecision(8) << r.generatedOverVendorRatio << ",\n";
        out << "      \"maxAbsError\": " << std::setprecision(8) << r.maxAbsError << ",\n";
        out << "      \"notes\": \"" << jsonEscape(r.notes) << "\",\n";
        out << "      \"kernelClass\": \"" << jsonEscape(r.kernelClass) << "\",\n";
        out << "      \"implementationKind\": \"" << jsonEscape(r.implementationKind) << "\",\n";
        out << "      \"tuningPath\": \"" << jsonEscape(r.tuningPath) << "\",\n";
        out << "      \"spmvRowNnzThreshold\": " << r.spmvRowNnzThreshold << ",\n";
        out << "      \"vendorDispatchSelected\": " << (r.vendorDispatchSelected ? "true" : "false");
        if (r.hostSubmitMeanMs > 0.0 || r.hostWallMeanMs > 0.0) {
            out << ",\n";
            out << "      \"hostSubmitMeanMs\": " << std::setprecision(8) << r.hostSubmitMeanMs << ",\n";
            out << "      \"hostWallMeanMs\": " << std::setprecision(8) << r.hostWallMeanMs << ",\n";
            out << "      \"hostSubmitOverGeneratedRatio\": " << std::setprecision(8)
                << (r.navatalaMs > 0.0 ? r.hostSubmitMeanMs / r.navatalaMs : 0.0) << ",\n";
            out << "      \"hostWallOverGeneratedRatio\": " << std::setprecision(8)
                << (r.navatalaMs > 0.0 ? r.hostWallMeanMs / r.navatalaMs : 0.0) << "\n";
        } else {
            out << "\n";
        }
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parseArgs(argc, argv);

    if (options.listOnly) {
        printBenchmarkList();
        return 0;
    }
    if (options.requireHipSparseLt && !NAVATALA_HAVE_HIPSPARSELT) {
        std::fprintf(stderr, "ERROR: --require-hipsparselt requested, but hipSPARSELt was not detected at build time\n");
        return 2;
    }
    if (options.includeHipSparseLtBenchmark && !NAVATALA_HAVE_HIPSPARSELT) {
        std::fprintf(stderr, "ERROR: --include-hipsparselt-benchmark requested, but hipSPARSELt was not detected at build time\n");
        return 2;
    }
#if !NAVATALA_HAVE_RUNTIME_WRAPPER
    if (options.includeWrapperGemmBenchmark) {
        std::fprintf(stderr, "ERROR: --include-wrapper-gemm-benchmark requested, but gpu_runtime was not linked into the benchmark\n");
        return 2;
    }
#endif
#if !NAVATALA_HAVE_MFMA_BENCHMARK
    if (options.includeMfmaBenchmark) {
        std::fprintf(stderr, "ERROR: --include-mfma-benchmark requested, but this binary was not built with NAVATALA_GPU_BUILD_MFMA_BENCHMARK=ON\n");
        return 2;
    }
#endif
    if (options.matrix == "wrapper-semantics" && !options.includeWrapperGemmBenchmark) {
        std::fprintf(stderr, "ERROR: --matrix wrapper-semantics requires --include-wrapper-gemm-benchmark\n");
        return 2;
    }

    int deviceCount = 0;
    check("hipGetDeviceCount", hipGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        std::fprintf(stderr, "ERROR: no HIP devices available\n");
        return 2;
    }
    check("hipSetDevice(0)", hipSetDevice(0));
    hipDeviceProp_t props{};
    check("hipGetDeviceProperties", hipGetDeviceProperties(&props, 0));
    std::printf("ROCm benchmark device: %s, gcnArch=%s, memory=%zu MiB\n",
                props.name,
                props.gcnArchName,
                static_cast<std::size_t>(props.totalGlobalMem / (1024 * 1024)));
    std::printf("hipSPARSELt detected at build time: %s\n",
                NAVATALA_HAVE_HIPSPARSELT ? "yes" : "no");
    std::printf("MFMA benchmark built: %s\n",
                NAVATALA_HAVE_MFMA_BENCHMARK ? "yes" : "no");

    rocblas_handle blas = nullptr;
    rocsparse_handle sparse = nullptr;
    check("rocblas_create_handle", rocblas_create_handle(&blas));
    check("rocsparse_create_handle", rocsparse_create_handle(&sparse));

    std::vector<Result> results;
    if (options.quick) {
        results.push_back(benchAxpy(options, blas, 1u << 16));
        results.push_back(benchGemm(options, blas, 128u, 128u, 128u));
        results.push_back(benchGemmF16PortableF32Out(options, blas, 128u, 128u, 128u));
#if NAVATALA_HAVE_MFMA_BENCHMARK
        if (options.includeMfmaBenchmark) {
            results.push_back(benchGemmF16Mfma(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64Direct(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64Shared(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64SharedEarlyBarrier(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64SharedPadded(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64Pipelined(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta128(options, blas, 128u, 128u, 128u));
        }
#endif
#if NAVATALA_HAVE_RUNTIME_WRAPPER
        if (options.includeWrapperGemmBenchmark) {
            results.push_back(benchGemmWrapperVendor(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 127u, 129u, 65u));
        }
#endif
        results.push_back(benchSpmv(options, props, sparse, 1u << 14, 7u));
#if NAVATALA_HAVE_HIPSPARSELT
        if (options.includeHipSparseLtBenchmark) {
            results.push_back(benchHipSparseLtStructuredGemmF16(options, blas, 128u, 128u, 128u));
        }
#endif
    } else if (options.matrix == "cfd-fixture") {
        const CsrFixture fixture = loadCsrFixture(options);
        results.push_back(benchCsrFixtureSpmv(options, props, sparse, fixture, false));
        results.push_back(benchCsrFixtureSpmv(options, props, sparse, fixture, true));
    } else if (options.matrix == "cta128-evidence") {
#if NAVATALA_HAVE_MFMA_BENCHMARK
        if (!options.includeMfmaBenchmark) {
            std::fprintf(stderr, "ERROR: --matrix cta128-evidence requires --include-mfma-benchmark\n");
            return 2;
        }
        const std::array<std::array<std::uint32_t, 3>, 6> ctaShapes{{
            {128u, 128u, 128u},
            {512u, 512u, 512u},
            {1024u, 1024u, 1024u},
            {1024u, 512u, 1024u},
            {512u, 1024u, 1024u},
            {2048u, 512u, 1024u},
        }};
        for (const auto& shape : ctaShapes) {
            const std::uint32_t m = shape[0];
            const std::uint32_t n = shape[1];
            const std::uint32_t k = shape[2];
            results.push_back(benchGemmF16PortableF32Out(options, blas, m, n, k));
            results.push_back(benchGemmF16MfmaCta64Shared(options, blas, m, n, k));
            results.push_back(benchGemmF16MfmaCta64Pipelined(options, blas, m, n, k));
            results.push_back(benchGemmF16MfmaCta128(options, blas, m, n, k));
            results.push_back(benchGemmF16MfmaEdge(options, blas, m, n, k, false));
            results.push_back(benchGemmF16MfmaEdge(options, blas, m, n, k, true));
#if NAVATALA_HAVE_RUNTIME_WRAPPER
            if (options.includeWrapperGemmBenchmark) {
                results.push_back(benchGemmF16F32WrapperMfma(options, blas, m, n, k));
            }
#endif
        }
#else
        std::fprintf(stderr, "ERROR: --matrix cta128-evidence requires NAVATALA_GPU_BUILD_MFMA_BENCHMARK=ON\n");
        return 2;
#endif
    } else if (options.matrix == "edge-tails") {
#if NAVATALA_HAVE_MFMA_BENCHMARK
        if (!options.includeMfmaBenchmark) {
            std::fprintf(stderr, "ERROR: --matrix edge-tails requires --include-mfma-benchmark\n");
            return 2;
        }
        const std::array<std::array<std::uint32_t, 3>, 3> edgeTailShapes{{
            {1023u, 1024u, 2003u},
            {1024u, 1023u, 2047u},
            {1025u, 513u, 1024u},
        }};
        for (const auto& shape : edgeTailShapes) {
            const std::uint32_t m = shape[0];
            const std::uint32_t n = shape[1];
            const std::uint32_t k = shape[2];
            const std::string suffix =
                "TAIL_M" + std::to_string(m) +
                "_N" + std::to_string(n) +
                "_K" + std::to_string(k);
            results.push_back(benchGemmF16MfmaEdge(options, blas, m, n, k, false));
            results.push_back(benchGemmF16MfmaEdge(options, blas, m, n, k, false, true));
            results.push_back(benchGemmF16MfmaEdge(options, blas, m, n, k, true));
#if NAVATALA_HAVE_RUNTIME_WRAPPER
            if (options.includeWrapperGemmBenchmark) {
                results.push_back(benchGemmF16F32WrapperMfma(
                    options, blas, m, n, k, suffix));
                results.push_back(benchGemmF16F32WrapperMfma(
                    options, blas, m, n, k, suffix,
                    1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
                    1u, "MFMA_EDGE_SPLIT", "mfma", false, "split"));
                results.push_back(benchGemmF16F32WrapperMfma(
                    options, blas, m, n, k, suffix,
                    1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
                    1u, "AUTO", "auto", true));
                results.push_back(benchGemmF16F32WrapperMfma(
                    options, blas, m, n, k, suffix,
                    1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
                    1u, "VENDOR", "vendor", true));
            }
#endif
        }
#else
        std::fprintf(stderr, "ERROR: --matrix edge-tails requires NAVATALA_GPU_BUILD_MFMA_BENCHMARK=ON\n");
        return 2;
#endif
    } else if (options.matrix == "wrapper-semantics") {
#if NAVATALA_HAVE_RUNTIME_WRAPPER
#if NAVATALA_HAVE_MFMA_BENCHMARK
        if (!options.includeMfmaBenchmark) {
            std::fprintf(stderr, "ERROR: --matrix wrapper-semantics requires --include-mfma-benchmark for direct semantic comparator rows\n");
            return 2;
        }
#endif
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 513u, 511u, 257u, "TAIL_SMALL"));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 1023u, 1024u, 2003u, "TAIL_LARGE"));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 1023u, 1024u, 2003u, "TAIL_LARGE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
            1u, "MFMA_EDGE_SPLIT", "mfma", false, "split"));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 1023u, 1024u, 2003u, "TAIL_LARGE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
            1u, "AUTO", "auto", true));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 1023u, 1024u, 2003u, "TAIL_LARGE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
            1u, "VENDOR", "vendor", true));
#if NAVATALA_HAVE_MFMA_BENCHMARK
        results.push_back(benchGemmF16MfmaSemantic(
            options, blas, 512u, 512u, 512u, false, "ALPHA_BETA",
            0.75f, 0.25f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE));
#endif
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 512u, 512u, 512u, "ALPHA_BETA",
            0.75f, 0.25f));
#if NAVATALA_HAVE_MFMA_BENCHMARK
        results.push_back(benchGemmF16MfmaSemantic(
            options, blas, 384u, 512u, 256u, false, "TRANSPOSE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_TRANSPOSE, NAVATALA_MATRIX_OP_TRANSPOSE));
#endif
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 384u, 512u, 256u, "TRANSPOSE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_TRANSPOSE, NAVATALA_MATRIX_OP_TRANSPOSE));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 384u, 512u, 256u, "TRANSPOSE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_TRANSPOSE, NAVATALA_MATRIX_OP_TRANSPOSE,
            1u, "AUTO", "auto", true));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 384u, 512u, 256u, "TRANSPOSE",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_TRANSPOSE, NAVATALA_MATRIX_OP_TRANSPOSE,
            1u, "VENDOR", "vendor", true));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 256u, 256u, 256u, "BATCHED",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE, 3u));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 256u, 256u, 256u, "BATCHED",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
            3u, "AUTO", "auto", true));
        results.push_back(benchGemmF16F32WrapperMfma(
            options, blas, 256u, 256u, 256u, "BATCHED",
            1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE,
            3u, "VENDOR", "vendor", true));
#else
        std::fprintf(stderr, "ERROR: --matrix wrapper-semantics requires NAVATALA_GPU_BUILD_BENCHMARKS with runtime wrapper support\n");
        return 2;
#endif
    } else if (options.matrix == "broad") {
        results.push_back(benchAxpy(options, blas, 1u << 16));
        results.push_back(benchAxpy(options, blas, 1u << 20));
        results.push_back(benchAxpy(options, blas, 1u << 22));
        results.push_back(benchGemm(options, blas, 128u, 128u, 128u));
        results.push_back(benchGemm(options, blas, 512u, 512u, 512u));
        results.push_back(benchGemm(options, blas, 1024u, 1024u, 1024u));
        results.push_back(benchGemmF16PortableF32Out(options, blas, 128u, 128u, 128u));
        results.push_back(benchGemmF16PortableF32Out(options, blas, 512u, 512u, 512u));
        results.push_back(benchGemmF16PortableF32Out(options, blas, 1024u, 1024u, 1024u));
#if NAVATALA_HAVE_MFMA_BENCHMARK
        if (options.includeMfmaBenchmark) {
            results.push_back(benchGemmF16Mfma(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16Mfma(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16Mfma(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16MfmaCta64Direct(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64Direct(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64Direct(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16MfmaCta64Shared(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64Shared(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64Shared(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16MfmaCta64SharedEarlyBarrier(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64SharedEarlyBarrier(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64SharedEarlyBarrier(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16MfmaCta64SharedPadded(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64SharedPadded(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64SharedPadded(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16MfmaCta64Pipelined(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta64Pipelined(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64Pipelined(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16MfmaCta128(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmF16MfmaCta128(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta128(options, blas, 1024u, 1024u, 1024u));
        }
#endif
#if NAVATALA_HAVE_RUNTIME_WRAPPER
        if (options.includeWrapperGemmBenchmark) {
            results.push_back(benchGemmWrapperVendor(options, blas, 128u, 128u, 128u));
            results.push_back(benchGemmWrapperVendor(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmWrapperVendor(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 127u, 129u, 65u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 1025u, 1023u, 257u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 1024u, 1024u, 1024u));
            results.push_back(benchGemmF16F32WrapperMfma(
                options, blas, 512u, 512u, 512u, "ALPHA_BETA",
                0.75f, 0.25f));
            results.push_back(benchGemmF16F32WrapperMfma(
                options, blas, 384u, 512u, 256u, "TRANSPOSE",
                1.0f, 0.0f, NAVATALA_MATRIX_OP_TRANSPOSE, NAVATALA_MATRIX_OP_TRANSPOSE));
            results.push_back(benchGemmF16F32WrapperMfma(
                options, blas, 256u, 256u, 256u, "BATCHED",
                1.0f, 0.0f, NAVATALA_MATRIX_OP_NONE, NAVATALA_MATRIX_OP_NONE, 3u));
        }
#endif
        results.push_back(benchSpmv(options, props, sparse, 1u << 14, 7u));
        results.push_back(benchSpmv(options, props, sparse, 1u << 18, 7u));
        results.push_back(benchSpmv(options, props, sparse, 1u << 20, 7u));
        results.push_back(benchSpmv(options, props, sparse, 1u << 18, 15u));
        results.push_back(benchSpmv(options, props, sparse, 1u << 18, 27u));
#if NAVATALA_HAVE_HIPSPARSELT
        if (options.includeHipSparseLtBenchmark) {
            results.push_back(benchHipSparseLtStructuredGemmF16(options, blas, 128u, 128u, 128u));
            results.push_back(benchHipSparseLtStructuredGemmF16(options, blas, 512u, 512u, 512u));
        }
#endif
    } else {
        results.push_back(benchAxpy(options, blas, 1u << 22));
        results.push_back(benchGemm(options, blas, 512u, 512u, 512u));
        results.push_back(benchGemmF16PortableF32Out(options, blas, 512u, 512u, 512u));
#if NAVATALA_HAVE_MFMA_BENCHMARK
        if (options.includeMfmaBenchmark) {
            results.push_back(benchGemmF16Mfma(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64Direct(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64Shared(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64SharedEarlyBarrier(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64SharedPadded(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta64Pipelined(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16MfmaCta128(options, blas, 512u, 512u, 512u));
        }
#endif
#if NAVATALA_HAVE_RUNTIME_WRAPPER
        if (options.includeWrapperGemmBenchmark) {
            results.push_back(benchGemmWrapperVendor(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 512u, 512u, 512u));
            results.push_back(benchGemmF16F32WrapperMfma(options, blas, 513u, 511u, 257u));
        }
#endif
        results.push_back(benchSpmv(options, props, sparse, 1u << 20, 7u));
#if NAVATALA_HAVE_HIPSPARSELT
        if (options.includeHipSparseLtBenchmark) {
            results.push_back(benchHipSparseLtStructuredGemmF16(options, blas, 512u, 512u, 512u));
        }
#endif
    }

    rocsparse_destroy_handle(sparse);
    rocblas_destroy_handle(blas);

    printResults(results);
    writeJson(options.jsonPath, options, props, results);

    const bool ok = std::all_of(results.begin(), results.end(), [](const Result& r) {
        return r.correctness;
    });
    return ok ? 0 : 1;
}
