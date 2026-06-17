// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Worked example: use the public Navatala GPU C ABI to compute
// y = alpha * x + y, then verify the result on the host.
//
// The example uses only navatala/navatala_ffi.h. It does not link against the
// internal C++ runtime API, so it remains valid when the public shared library
// hides internal implementation symbols.

#include "navatala/navatala_ffi.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

const char* statusName(NavatalaErrorCode code) {
    switch (code) {
        case NAVATALA_SUCCESS: return "NAVATALA_SUCCESS";
        case NAVATALA_INVALID_HANDLE: return "NAVATALA_INVALID_HANDLE";
        case NAVATALA_INVALID_PARAM: return "NAVATALA_INVALID_PARAM";
        case NAVATALA_GPU_ERROR: return "NAVATALA_GPU_ERROR";
        case NAVATALA_OUT_OF_MEMORY: return "NAVATALA_OUT_OF_MEMORY";
        case NAVATALA_NOT_IMPLEMENTED: return "NAVATALA_NOT_IMPLEMENTED";
        case NAVATALA_IO_ERROR: return "NAVATALA_IO_ERROR";
        case NAVATALA_RUNTIME_ERROR: return "NAVATALA_RUNTIME_ERROR";
        case NAVATALA_OVERFLOW_ERROR: return "NAVATALA_OVERFLOW_ERROR";
        case NAVATALA_TIMEOUT: return "NAVATALA_TIMEOUT";
        case NAVATALA_CANCELLED: return "NAVATALA_CANCELLED";
        case NAVATALA_NOT_FOUND: return "NAVATALA_NOT_FOUND";
        case NAVATALA_ALREADY_EXISTS: return "NAVATALA_ALREADY_EXISTS";
        case NAVATALA_PERMISSION_DENIED: return "NAVATALA_PERMISSION_DENIED";
        case NAVATALA_INTERNAL_ERROR: return "NAVATALA_INTERNAL_ERROR";
        default: return "NAVATALA_UNKNOWN_ERROR";
    }
}

bool check(NavatalaErrorCode code, const char* label) {
    if (code == NAVATALA_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ERROR: %s failed: %s (%d)\n",
                 label, statusName(code), static_cast<int>(code));
    return false;
}

}  // namespace

int main() {
    constexpr std::uint32_t N = 1024;
    constexpr float kAlpha = 2.0f;

    NavatalaGpuContext* ctx = nullptr;
    NavatalaErrorCode status =
        navatala_gpu_create_context(NAVATALA_BACKEND_AUTO_FFI, 0, &ctx);
    if (status != NAVATALA_SUCCESS) {
        std::printf("[skip] no Navatala GPU backend/context available: %s (%d)\n",
                    statusName(status), static_cast<int>(status));
        return 0;
    }

    NavatalaGpuQueue* queue = nullptr;
    NavatalaGpuBuffer* x = nullptr;
    NavatalaGpuBuffer* y = nullptr;

    if (!check(navatala_gpu_create_queue(ctx, -1, &queue), "create_queue") ||
        !check(navatala_gpu_create_buffer(ctx, N * sizeof(float), &x), "create_buffer(x)") ||
        !check(navatala_gpu_create_buffer(ctx, N * sizeof(float), &y), "create_buffer(y)")) {
        navatala_gpu_destroy_buffer(y);
        navatala_gpu_destroy_buffer(x);
        navatala_gpu_destroy_queue(queue);
        navatala_gpu_destroy_context(ctx);
        return 1;
    }

    std::vector<float> hostX(N);
    std::vector<float> hostY(N, 1.0f);
    for (std::uint32_t i = 0; i < N; ++i) {
        hostX[i] = static_cast<float>(i);
    }

    if (!check(navatala_gpu_copy_h2d(x, hostX.data(), N * sizeof(float), queue), "copy_h2d(x)") ||
        !check(navatala_gpu_copy_h2d(y, hostY.data(), N * sizeof(float), queue), "copy_h2d(y)") ||
        !check(navatala_gpu_axpy_f32(x, y, kAlpha, N, queue), "axpy_f32") ||
        !check(navatala_gpu_queue_sync(queue), "queue_sync") ||
        !check(navatala_gpu_copy_d2h(y, hostY.data(), N * sizeof(float), queue), "copy_d2h(y)")) {
        navatala_gpu_destroy_buffer(y);
        navatala_gpu_destroy_buffer(x);
        navatala_gpu_destroy_queue(queue);
        navatala_gpu_destroy_context(ctx);
        return 1;
    }

    int errors = 0;
    for (std::uint32_t i = 0; i < N; ++i) {
        const float expected = 1.0f + kAlpha * static_cast<float>(i);
        if (std::fabs(hostY[i] - expected) > 1.0e-5f) {
            if (errors < 8) {
                std::fprintf(stderr,
                             "MISMATCH at %u: got %g, expected %g\n",
                             i, static_cast<double>(hostY[i]),
                             static_cast<double>(expected));
            }
            ++errors;
        }
    }

    navatala_gpu_destroy_buffer(y);
    navatala_gpu_destroy_buffer(x);
    navatala_gpu_destroy_queue(queue);
    navatala_gpu_destroy_context(ctx);

    if (errors != 0) {
        std::fprintf(stderr, "FAIL: %d / %u mismatches\n", errors, N);
        return 1;
    }

    std::printf("PASS: y = y + %g * x verified for N=%u through navatala_ffi.h\n",
                static_cast<double>(kAlpha), N);
    return 0;
}
