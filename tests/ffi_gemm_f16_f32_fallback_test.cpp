// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

#include <navatala/navatala_ffi.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int set_env_value(const char* key, const char* value) {
#if defined(_WIN32)
    return _putenv_s(key, value);
#else
    return setenv(key, value, 1);
#endif
}

int unset_env_value(const char* key) {
#if defined(_WIN32)
    return _putenv_s(key, "");
#else
    return unsetenv(key);
#endif
}

struct Runtime {
    NavatalaGpuContext* ctx = nullptr;
    NavatalaGpuQueue* queue = nullptr;
    NavatalaBackend backend = NAVATALA_BACKEND_AUTO_FFI;

    Runtime() {
        const NavatalaBackend requested = requested_backend();
        if (navatala_gpu_create_context(requested, 0, &ctx) != NAVATALA_SUCCESS) {
            std::cerr << "failed to create context for backend " << backend_name(requested) << "\n";
            std::exit(2);
        }
        backend = navatala_gpu_context_get_backend(ctx);
        if (navatala_gpu_create_queue(ctx, -1, &queue) != NAVATALA_SUCCESS) {
            std::cerr << "failed to create queue for backend " << backend_name(backend) << "\n";
            std::exit(2);
        }
    }

    ~Runtime() {
        navatala_gpu_destroy_queue(queue);
        navatala_gpu_destroy_context(ctx);
    }

    static const char* backend_name(NavatalaBackend value) {
        switch (value) {
            case NAVATALA_BACKEND_CUDA_FFI: return "cuda";
            case NAVATALA_BACKEND_HIP_FFI: return "hip";
            case NAVATALA_BACKEND_METAL_FFI: return "metal";
            case NAVATALA_BACKEND_OPENCL_FFI: return "opencl";
            case NAVATALA_BACKEND_VULKAN_FFI: return "vulkan";
            case NAVATALA_BACKEND_AUTO_FFI: return "auto";
            default: return "unknown";
        }
    }

    static NavatalaBackend requested_backend() {
        const char* value = std::getenv("NAVATALA_GPU_TEST_BACKEND");
        if (!value || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
            return NAVATALA_BACKEND_AUTO_FFI;
        }
        if (std::strcmp(value, "cuda") == 0) return NAVATALA_BACKEND_CUDA_FFI;
        if (std::strcmp(value, "hip") == 0 || std::strcmp(value, "rocm") == 0) {
            return NAVATALA_BACKEND_HIP_FFI;
        }
        if (std::strcmp(value, "metal") == 0) return NAVATALA_BACKEND_METAL_FFI;
        if (std::strcmp(value, "opencl") == 0) return NAVATALA_BACKEND_OPENCL_FFI;
        if (std::strcmp(value, "vulkan") == 0) return NAVATALA_BACKEND_VULKAN_FFI;
        std::cerr << "unknown NAVATALA_GPU_TEST_BACKEND=" << value << "\n";
        std::exit(2);
    }
};

struct EnvOverride {
    const char* name;
    std::string oldValue;
    bool hadOldValue;

    EnvOverride(const char* key, const char* value)
        : name(key)
    {
        const char* old = std::getenv(key);
        hadOldValue = old != nullptr;
        oldValue = old ? old : "";
        if (set_env_value(name, value) != 0) {
            std::cerr << "failed to set " << name << "\n";
            std::exit(2);
        }
    }

    ~EnvOverride() {
        if (hadOldValue) {
            set_env_value(name, oldValue.c_str());
        } else {
            unset_env_value(name);
        }
    }
};

struct Buffer {
    NavatalaGpuBuffer* handle = nullptr;

    Buffer(NavatalaGpuContext* ctx, size_t bytes) {
        if (navatala_gpu_create_buffer(ctx, bytes, &handle) != NAVATALA_SUCCESS) {
            std::cerr << "failed to create buffer\n";
            std::exit(2);
        }
    }

    ~Buffer() {
        navatala_gpu_destroy_buffer(handle);
    }
};

std::uint16_t h(unsigned value) {
    static const std::uint16_t lut[] = {
        0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600,
        0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00
    };
    return value < (sizeof(lut) / sizeof(lut[0])) ? lut[value] : 0;
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 1.0e-4f;
}

void upload(NavatalaGpuBuffer* buffer, const void* data, size_t bytes, NavatalaGpuQueue* queue) {
    if (navatala_gpu_copy_h2d(buffer, data, bytes, queue) != NAVATALA_SUCCESS) {
        std::cerr << "copy_h2d failed\n";
        std::exit(2);
    }
}

void download(const NavatalaGpuBuffer* buffer, void* data, size_t bytes, NavatalaGpuQueue* queue) {
    if (navatala_gpu_copy_d2h(buffer, data, bytes, queue) != NAVATALA_SUCCESS) {
        std::cerr << "copy_d2h failed\n";
        std::exit(2);
    }
}

void test_transposed_alpha_beta(Runtime& rt) {
    // Logical A = [[1,2,3],[4,5,6]], stored transposed as [3,2].
    const std::vector<std::uint16_t> a = {
        h(1), h(4),
        h(2), h(5),
        h(3), h(6)
    };
    // Logical B = [[7,8],[9,10],[11,12]], stored transposed as [2,3].
    const std::vector<std::uint16_t> b = {
        h(7), h(9), h(11),
        h(8), h(10), h(12)
    };
    std::vector<float> c = {1.0f, 1.0f, 1.0f, 1.0f};
    Buffer db_a(rt.ctx, a.size() * sizeof(std::uint16_t));
    Buffer db_b(rt.ctx, b.size() * sizeof(std::uint16_t));
    Buffer db_c(rt.ctx, c.size() * sizeof(float));
    upload(db_a.handle, a.data(), a.size() * sizeof(std::uint16_t), rt.queue);
    upload(db_b.handle, b.data(), b.size() * sizeof(std::uint16_t), rt.queue);
    upload(db_c.handle, c.data(), c.size() * sizeof(float), rt.queue);

    const NavatalaErrorCode status = navatala_gpu_gemm_f16_f32_ex(
        db_a.handle, db_b.handle, db_c.handle,
        2, 2, 3, 0.5f, 2.0f,
        NAVATALA_MATRIX_OP_TRANSPOSE,
        NAVATALA_MATRIX_OP_TRANSPOSE,
        rt.queue);
    if (status != NAVATALA_SUCCESS) {
        std::cerr << "gemm_f16_f32_ex failed: " << status << "\n";
        std::exit(1);
    }

    download(db_c.handle, c.data(), c.size() * sizeof(float), rt.queue);
    const float expected[] = {31.0f, 34.0f, 71.5f, 79.0f};
    for (size_t i = 0; i < c.size(); ++i) {
        if (!near(c[i], expected[i])) {
            std::cerr << "transpose result mismatch at " << i
                      << ": got " << c[i] << ", expected " << expected[i] << "\n";
            std::exit(1);
        }
    }
}

void test_strided_batched(Runtime& rt) {
    const std::vector<std::uint16_t> a = {h(2), 0, h(3)};
    const std::vector<std::uint16_t> b = {h(4), 0, h(5)};
    std::vector<float> c = {1.0f, -99.0f, 1.0f};
    Buffer db_a(rt.ctx, a.size() * sizeof(std::uint16_t));
    Buffer db_b(rt.ctx, b.size() * sizeof(std::uint16_t));
    Buffer db_c(rt.ctx, c.size() * sizeof(float));
    upload(db_a.handle, a.data(), a.size() * sizeof(std::uint16_t), rt.queue);
    upload(db_b.handle, b.data(), b.size() * sizeof(std::uint16_t), rt.queue);
    upload(db_c.handle, c.data(), c.size() * sizeof(float), rt.queue);

    const NavatalaErrorCode status = navatala_gpu_gemm_f16_f32_strided_batched(
        db_a.handle, db_b.handle, db_c.handle,
        1, 1, 1, 1.0f, 1.0f,
        NAVATALA_MATRIX_OP_NONE,
        NAVATALA_MATRIX_OP_NONE,
        2, 2, 2, 2,
        rt.queue);
    if (status != NAVATALA_SUCCESS) {
        std::cerr << "strided batched GEMM failed: " << status << "\n";
        std::exit(1);
    }

    download(db_c.handle, c.data(), c.size() * sizeof(float), rt.queue);
    if (!near(c[0], 9.0f) || !near(c[1], -99.0f) || !near(c[2], 16.0f)) {
        std::cerr << "batched result mismatch: "
                  << c[0] << ", " << c[1] << ", " << c[2] << "\n";
        std::exit(1);
    }
}

void test_broadcast_input_batches(Runtime& rt) {
    const std::vector<std::uint16_t> a = {h(2)};
    const std::vector<std::uint16_t> b = {h(4), h(5)};
    std::vector<float> c = {1.0f, 1.0f};
    Buffer db_a(rt.ctx, a.size() * sizeof(std::uint16_t));
    Buffer db_b(rt.ctx, b.size() * sizeof(std::uint16_t));
    Buffer db_c(rt.ctx, c.size() * sizeof(float));
    upload(db_a.handle, a.data(), a.size() * sizeof(std::uint16_t), rt.queue);
    upload(db_b.handle, b.data(), b.size() * sizeof(std::uint16_t), rt.queue);
    upload(db_c.handle, c.data(), c.size() * sizeof(float), rt.queue);

    const NavatalaErrorCode status = navatala_gpu_gemm_f16_f32_strided_batched(
        db_a.handle, db_b.handle, db_c.handle,
        1, 1, 1, 1.0f, 1.0f,
        NAVATALA_MATRIX_OP_NONE,
        NAVATALA_MATRIX_OP_NONE,
        0, 1, 1, 2,
        rt.queue);
    if (status != NAVATALA_SUCCESS) {
        std::cerr << "broadcast input batched GEMM failed: " << status << "\n";
        std::exit(1);
    }

    download(db_c.handle, c.data(), c.size() * sizeof(float), rt.queue);
    if (!near(c[0], 9.0f) || !near(c[1], 11.0f)) {
        std::cerr << "broadcast input result mismatch: "
                  << c[0] << ", " << c[1] << "\n";
        std::exit(1);
    }

    const NavatalaErrorCode overlapping_c = navatala_gpu_gemm_f16_f32_strided_batched(
        db_a.handle, db_b.handle, db_c.handle,
        1, 1, 1, 1.0f, 1.0f,
        NAVATALA_MATRIX_OP_NONE,
        NAVATALA_MATRIX_OP_NONE,
        0, 1, 0, 2,
        rt.queue);
    if (overlapping_c != NAVATALA_OVERFLOW_ERROR) {
        std::cerr << "overlapping C batch stride should be rejected, got "
                  << overlapping_c << "\n";
        std::exit(1);
    }
}

void run_suite(Runtime& rt, const char* label) {
    std::cout << "running F16/F32 GEMM FFI suite: " << label
              << " backend=" << Runtime::backend_name(rt.backend) << "\n";
    test_transposed_alpha_beta(rt);
    test_strided_batched(rt);
    test_broadcast_input_batches(rt);
}

} // namespace

int main() {
    Runtime rt;
    run_suite(rt, "default");
    if (rt.backend == NAVATALA_BACKEND_HIP_FFI) {
        EnvOverride impl("NAVATALA_GPU_GEMM_IMPL", "mfma");
        {
            EnvOverride mode("NAVATALA_GPU_GEMM_MFMA_MODE", "cta64");
            run_suite(rt, "forced-mfma-cta64");
        }
        {
            EnvOverride mode("NAVATALA_GPU_GEMM_MFMA_MODE", "cta128");
            run_suite(rt, "forced-mfma-cta128");
        }
    }
    return 0;
}
