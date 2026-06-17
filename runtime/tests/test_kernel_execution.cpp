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

// GPU Runtime - End-to-End Kernel Execution Tests
// Test actual kernel compilation, execution, and result validation

#include "../include/gpu_runtime.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <functional>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using namespace GpuRuntime;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Test kernel definition with source code and validation
struct TestKernel {
    std::string name;
    std::string source;           // OpenCL C, SPIR-V assembly, etc.
    std::string entryPoint;
    std::vector<std::pair<std::string, size_t>> inputNames;  // name, size in bytes
    std::vector<std::pair<std::string, size_t>> outputNames;
    // Called while inputs are mapped (MapMode::Write), before kernel launch.
    std::function<void(const std::vector<void*>&)> initializeInputs;
    // Called while outputs are mapped (MapMode::Read), after queue->synchronize().
    std::function<void(const std::vector<void*>&)> validateOutputs;
};

// ============================================================================
// Helper: Compile CUDA C++ kernel source to PTX (requires nvcc on PATH)
// ============================================================================

static bool writeFile(const std::string& path, const std::string& contents, std::string* err) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (err) *err = "Failed to open file for write: " + path;
        return false;
    }
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!out.good()) {
        if (err) *err = "Failed to write file: " + path;
        return false;
    }
    return true;
}

static bool readFileBytes(const std::string& path, std::vector<std::uint8_t>* out, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "Failed to open file for read: " + path;
        return false;
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        if (err) *err = "Read empty file: " + path;
        return false;
    }
    *out = std::move(bytes);
    return true;
}

static bool compileCudaToPtx(const std::string& cudaSrc, std::vector<std::uint8_t>* outPtx, std::string* err) {
    // Create temp paths.
    char cuPath[] = "/tmp/gpu_runtime_kernel_XXXXXX.cu";
    int fd = ::mkstemps(cuPath, 3); // ".cu" suffix length = 3
    if (fd < 0) {
        if (err) *err = "mkstemps failed creating temp .cu file";
        return false;
    }
    ::close(fd);
    std::string cu = cuPath;
    std::string ptx = cu.substr(0, cu.size() - 3) + ".ptx";

    if (!writeFile(cu, cudaSrc, err)) {
        std::remove(cu.c_str());
        return false;
    }

    // Compile to PTX. Keep it simple and portable.
    std::ostringstream cmd;
    cmd << "nvcc --ptx -O2 -std=c++14 " << cu << " -o " << ptx << " 2>/tmp/gpu_runtime_nvcc_err.txt";
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::string nvccErr;
        std::vector<std::uint8_t> nvccErrBytes;
        if (readFileBytes("/tmp/gpu_runtime_nvcc_err.txt", &nvccErrBytes, nullptr)) {
            nvccErr.assign(reinterpret_cast<const char*>(nvccErrBytes.data()), nvccErrBytes.size());
        }
        if (err) *err = "nvcc failed (" + std::to_string(rc) + "): " + nvccErr;
        std::remove(cu.c_str());
        std::remove(ptx.c_str());
        return false;
    }

    bool ok = readFileBytes(ptx, outPtx, err);
    std::remove(cu.c_str());
    std::remove(ptx.c_str());
    return ok;
}

// ============================================================================
// Helper: Compile HIP C++ kernel source to HSACO (requires hipcc on PATH)
// ============================================================================

static bool compileHipToHsaco(const std::string& hipSrc, const std::string& arch, std::vector<std::uint8_t>* outHsaco, std::string* err) {
    if (!outHsaco) return false;
    outHsaco->clear();

    if (arch.empty()) {
        if (err) *err = "HIP arch is empty (expected e.g. gfx942)";
        return false;
    }

    char hipPath[] = "/tmp/gpu_runtime_hip_kernel_XXXXXX.hip";
    int fd = ::mkstemps(hipPath, 4); // ".hip" suffix length = 4
    if (fd < 0) {
        if (err) *err = "mkstemps failed creating temp .hip file";
        return false;
    }
    ::close(fd);
    std::string hip = hipPath;
    std::string hsaco = hip.substr(0, hip.size() - 4) + ".hsaco";

    if (!writeFile(hip, hipSrc, err)) {
        std::remove(hip.c_str());
        return false;
    }

    std::ostringstream cmd;
    cmd << "hipcc --genco -O2 --offload-arch=" << arch << " " << hip << " -o " << hsaco
        << " 2>/tmp/gpu_runtime_hipcc_err.txt";
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::string hipccErr;
        std::vector<std::uint8_t> hipccErrBytes;
        if (readFileBytes("/tmp/gpu_runtime_hipcc_err.txt", &hipccErrBytes, nullptr)) {
            hipccErr.assign(reinterpret_cast<const char*>(hipccErrBytes.data()), hipccErrBytes.size());
        }
        if (err) *err = "hipcc failed (" + std::to_string(rc) + "): " + hipccErr;
        std::remove(hip.c_str());
        std::remove(hsaco.c_str());
        return false;
    }

    bool ok = readFileBytes(hsaco, outHsaco, err);
    std::remove(hip.c_str());
    std::remove(hsaco.c_str());
    return ok;
}

/// Check if a specific backend is available on this system
bool isBackendAvailable(const std::string& backend) {
    // Try to create a device with the requested backend
    // Set environment variable to force backend selection
    const char* originalBackend = std::getenv("GPU_RUNTIME_BACKEND");

    // Temporarily set backend
    if (backend == "opencl") {
        setenv("GPU_RUNTIME_BACKEND", "opencl", 1);
    } else if (backend == "vulkan") {
        setenv("GPU_RUNTIME_BACKEND", "vulkan", 1);
    } else if (backend == "cuda") {
        setenv("GPU_RUNTIME_BACKEND", "cuda", 1);
    } else if (backend == "hip") {
        setenv("GPU_RUNTIME_BACKEND", "hip", 1);
    } else if (backend == "metal") {
        setenv("GPU_RUNTIME_BACKEND", "metal", 1);
    }

    auto device = Device::create(0);

    // Restore original backend setting
    if (originalBackend) {
        setenv("GPU_RUNTIME_BACKEND", originalBackend, 1);
    } else {
        unsetenv("GPU_RUNTIME_BACKEND");
    }

    return device != nullptr;
}

/// Run a single kernel test
bool runKernelTest(Device& device, const std::string& backend, const TestKernel& test, std::string* outError = nullptr) {
    std::cout << "=== Running: " << test.name << " (" << backend << ") ===" << std::endl;

    try {
        // Create program from source
        ProgramSource programSrc;
        if (backend == "opencl") {
            programSrc.kind = ProgramSource::Kind::OpenClC;
        } else if (backend == "vulkan") {
            programSrc.kind = ProgramSource::Kind::Spirv;
        } else if (backend == "cuda") {
            programSrc.kind = ProgramSource::Kind::Ptx;
        } else if (backend == "hip") {
            programSrc.kind = ProgramSource::Kind::Hsaco;
        } else if (backend == "metal") {
            programSrc.kind = ProgramSource::Kind::Msl;
        } else {
            std::cout << "FAIL: Unknown backend: " << backend << std::endl;
            return false;
        }

        programSrc.entryPoint = test.entryPoint;
        if (backend == "cuda") {
            std::vector<std::uint8_t> ptxBytes;
            std::string compileErr;
            if (!compileCudaToPtx(test.source, &ptxBytes, &compileErr)) {
                throw std::runtime_error("Failed to compile PTX via nvcc: " + compileErr);
            }
            programSrc.bytes = std::move(ptxBytes);
        } else if (backend == "hip") {
            std::string arch = device.getComputeCapability();
            // HIP reports strings like "gfx942:sramecc+:xnack-" on some systems; hipcc expects "gfx942".
            if (auto colon = arch.find(':'); colon != std::string::npos) {
                arch = arch.substr(0, colon);
            }
            if (const char* overrideArch = std::getenv("GPU_HIP_ARCH")) {
                if (overrideArch[0] != '\0') {
                    arch = overrideArch;
                }
            }
            std::vector<std::uint8_t> hsacoBytes;
            std::string compileErr;
            if (!compileHipToHsaco(test.source, arch, &hsacoBytes, &compileErr)) {
                throw std::runtime_error("Failed to compile HSACO via hipcc: " + compileErr);
            }
            programSrc.bytes = std::move(hsacoBytes);
        } else {
            programSrc.bytes.assign(test.source.begin(), test.source.end());
        }

        auto program = device.createProgram(programSrc);
        if (!program) {
            std::cout << "FAIL: Failed to create program from source" << std::endl;
            return false;
        }

        // Create queue
        auto queue = device.createQueue(StreamPriority::Normal);
        if (!queue) {
            std::cout << "FAIL: Failed to create queue" << std::endl;
            return false;
        }

        // Allocate and initialize input/output buffers
        std::vector<std::unique_ptr<Buffer>> inputBuffers;
        std::vector<std::unique_ptr<Buffer>> outputBuffers;
        std::vector<void*> allKernelArgs;

        for (const auto& [name, size] : test.inputNames) {
            auto buffer = device.createBuffer(size, MemoryKind::HostPinned);
            if (!buffer) {
                std::cout << "FAIL: Failed to create input buffer" << std::endl;
                return false;
            }
            void* hostPtr = buffer->mapPtr(MapMode::Write);
            if (!hostPtr) {
                std::cout << "FAIL: Failed to map input buffer" << std::endl;
                return false;
            }
            inputBuffers.push_back(std::move(buffer));
        }

        for (const auto& [name, size] : test.outputNames) {
            auto buffer = device.createBuffer(size, MemoryKind::HostPinned);
            if (!buffer) {
                std::cout << "FAIL: Failed to create output buffer" << std::endl;
                return false;
            }
            void* hostPtr = buffer->mapPtr(MapMode::ReadWrite);
            if (!hostPtr) {
                std::cout << "FAIL: Failed to map output buffer" << std::endl;
                return false;
            }
            // Initialize output to zero
            std::memset(hostPtr, 0, size);
            buffer->unmap();
            outputBuffers.push_back(std::move(buffer));
        }

        // Initialize inputs (while mapped)
        if (test.initializeInputs) {
            std::vector<void*> inputPtrs;
            inputPtrs.reserve(inputBuffers.size());
            for (auto& buf : inputBuffers) {
                inputPtrs.push_back(buf->getHostPointer());
            }
            test.initializeInputs(inputPtrs);
        }

        // Unmap buffers before kernel launch
        for (size_t i = 0; i < inputBuffers.size(); ++i) {
            inputBuffers[i]->unmap();
        }

        // Collect kernel arguments (all buffers)
        allKernelArgs.clear();
        for (auto& buf : inputBuffers) {
            allKernelArgs.push_back(buf.get());
        }
        for (auto& buf : outputBuffers) {
            allKernelArgs.push_back(buf.get());
        }

        // Launch kernel with default dispatch parameters
        // Using 256 workgroup size and reasonable grid size
        const uint32_t blockSize = 256;
        const uint32_t gridSize = 4;  // 1024 elements total

        std::vector<Buffer*> kernelArgs;
        for (auto* arg : allKernelArgs) {
            kernelArgs.push_back(static_cast<Buffer*>(arg));
        }

        queue->submit(*program, kernelArgs, gridSize, 1, 1, blockSize, 1, 1);
        queue->synchronize();

        // Map output buffers back to host
        std::vector<void*> outputResultPtrs;
        for (size_t i = 0; i < outputBuffers.size(); ++i) {
            void* hostPtr = outputBuffers[i]->mapPtr(MapMode::Read);
            if (!hostPtr) {
                std::cout << "FAIL: Failed to map output buffer for reading" << std::endl;
                return false;
            }
            outputResultPtrs.push_back(hostPtr);
        }

        // Run validation
        if (test.validateOutputs) {
            test.validateOutputs(outputResultPtrs);
        }

        // Cleanup
        for (size_t i = 0; i < outputBuffers.size(); ++i) {
            outputBuffers[i]->unmap();
        }

        std::cout << "PASS: " << test.name << std::endl;
        return true;

    } catch (const std::exception& e) {
        if (outError) {
            *outError = e.what();
        }
        std::cout << "FAIL: Exception - " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Test Kernels
// ============================================================================

// OpenCL Square Kernel
const char* squareKernelOpenCL = R"(
__kernel void square_kernel(__global const float* input, __global float* output) {
    int gid = get_global_id(0);
    float value = input[gid];
    output[gid] = value * value;
}
)";

// SAXPY OpenCL Kernel: y = alpha * x + y
const char* saxpyKernelOpenCL = R"(
__kernel void saxpy_kernel(__global const float* alpha, __global const float* x, __global const float* y, __global float* result) {
    int gid = get_global_id(0);
    float a = alpha[0];
    result[gid] = a * x[gid] + y[gid];
}
)";

// CUDA Square Kernel (compiled to PTX via nvcc)
const char* squareKernelCuda = R"(
extern "C" __global__ void square_kernel(const float* input, float* output) {
    int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    float value = input[gid];
    output[gid] = value * value;
}
)";

// CUDA SAXPY Kernel: result = alpha * x + y
const char* saxpyKernelCuda = R"(
extern "C" __global__ void saxpy_kernel(const float* alpha, const float* x, const float* y, float* result) {
    int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    float a = alpha[0];
    result[gid] = a * x[gid] + y[gid];
}
)";

// HIP Square Kernel (compiled to HSACO via hipcc)
const char* squareKernelHip = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void square_kernel(const float* input, float* output) {
    int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    float value = input[gid];
    output[gid] = value * value;
}
)";

// HIP SAXPY Kernel: result = alpha * x + y
const char* saxpyKernelHip = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void saxpy_kernel(const float* alpha, const float* x, const float* y, float* result) {
    int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    float a = alpha[0];
    result[gid] = a * x[gid] + y[gid];
}
)";

// Metal Square Kernel (MSL)
const char* squareKernelMsl = R"(
#include <metal_stdlib>
using namespace metal;
kernel void square_kernel(const device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    float v = input[gid];
    output[gid] = v * v;
}
)";

// Metal SAXPY Kernel (MSL): result = alpha * x + y
const char* saxpyKernelMsl = R"(
#include <metal_stdlib>
using namespace metal;
kernel void saxpy_kernel(const device float* alpha [[buffer(0)]],
                         const device float* x [[buffer(1)]],
                         const device float* y [[buffer(2)]],
                         device float* result [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    float a = alpha[0];
    result[gid] = a * x[gid] + y[gid];
}
)";

// ============================================================================
// Test Definitions
// ============================================================================

void test_square_opencl() {
    std::cout << "\n=== Test: Square (OpenCL) ===" << std::endl;

    if (!isBackendAvailable("opencl")) {
        std::cout << "SKIP: OpenCL backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "opencl", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create OpenCL device" << std::endl;
        return;
    }

    // Prepare test data: [1.0, 2.0, 3.0, 4.0, ...]
    std::vector<float> input(1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    TestKernel test;
    test.name = "Square";
    test.source = squareKernelOpenCL;
    test.entryPoint = "square_kernel";
    test.inputNames = {{"input", input.size() * sizeof(float)}};
    test.outputNames = {{"output", input.size() * sizeof(float)}};
    test.initializeInputs = [&input](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], input.data(), input.size() * sizeof(float));
    };
    test.validateOutputs = [&input](const std::vector<void*>& outputPtrs) {
        const float* outputData = static_cast<const float*>(outputPtrs[0]);

        // Verify first 16 elements
        for (size_t i = 0; i < 16; ++i) {
            float expected = input[i] * input[i];
            float actual = outputData[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    // Initialize input buffer with test data
    std::string err;
    bool success = runKernelTest(*device, "opencl", test, &err);
    if (!success && err.find("/usr/lib/clc/") != std::string::npos) {
        std::cout << "SKIP: OpenCL toolchain missing libclc bitcode: " << err << std::endl;
        return;
    }
    assert(success && "Square test should pass");
}

void test_saxpy_opencl() {
    std::cout << "\n=== Test: SAXPY (OpenCL) ===" << std::endl;

    if (!isBackendAvailable("opencl")) {
        std::cout << "SKIP: OpenCL backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "opencl", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create OpenCL device" << std::endl;
        return;
    }

    // Prepare test data: alpha=2.0, x=[1,2,3,...], y=[0.1, 0.2, 0.3, ...]
    float alpha = 2.0f;
    std::vector<float> x(1024);
    std::vector<float> y(1024);
    for (size_t i = 0; i < 1024; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>(i + 1) * 0.1f;
    }

    TestKernel test;
    test.name = "SAXPY";
    test.source = saxpyKernelOpenCL;
    test.entryPoint = "saxpy_kernel";
    test.inputNames = {
        {"alpha", sizeof(float)},
        {"x", x.size() * sizeof(float)},
        {"y", y.size() * sizeof(float)}
    };
    test.outputNames = {{"result", 1024 * sizeof(float)}};
    test.initializeInputs = [&alpha, &x, &y](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], &alpha, sizeof(float));
        std::memcpy(inputPtrs[1], x.data(), x.size() * sizeof(float));
        std::memcpy(inputPtrs[2], y.data(), y.size() * sizeof(float));
    };
    test.validateOutputs = [&alpha, &x, &y](const std::vector<void*>& outputPtrs) {
        const float* resultPtr = static_cast<const float*>(outputPtrs[0]);

        // Verify first 16 elements: result = alpha * x + y
        for (size_t i = 0; i < 16; ++i) {
            float expected = alpha * x[i] + y[i];
            float actual = resultPtr[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "opencl", test, &err);
    if (!success && err.find("/usr/lib/clc/") != std::string::npos) {
        std::cout << "SKIP: OpenCL toolchain missing libclc bitcode: " << err << std::endl;
        return;
    }
    assert(success && "SAXPY test should pass");
}

void test_square_vulkan() {
    std::cout << "\n=== Test: Square (Vulkan) ===" << std::endl;

    if (!isBackendAvailable("vulkan")) {
        std::cout << "SKIP: Vulkan backend not available" << std::endl;
        return;
    }

    std::cout << "INFO: Vulkan SPIR-V kernel testing not yet implemented" << std::endl;
    std::cout << "SKIP: Need SPIR-V compiler integration" << std::endl;
}

void test_square_cuda() {
    std::cout << "\n=== Test: Square (CUDA) ===" << std::endl;

    if (!isBackendAvailable("cuda")) {
        std::cout << "SKIP: CUDA backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "cuda", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create CUDA device" << std::endl;
        return;
    }

    std::vector<float> input(1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    TestKernel test;
    test.name = "Square";
    test.source = squareKernelCuda;
    test.entryPoint = "square_kernel";
    test.inputNames = {{"input", input.size() * sizeof(float)}};
    test.outputNames = {{"output", input.size() * sizeof(float)}};
    test.initializeInputs = [&input](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], input.data(), input.size() * sizeof(float));
    };
    test.validateOutputs = [&input](const std::vector<void*>& outputPtrs) {
        const float* outputData = static_cast<const float*>(outputPtrs[0]);
        for (size_t i = 0; i < 16; ++i) {
            float expected = input[i] * input[i];
            float actual = outputData[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "cuda", test, &err);
    assert(success && "CUDA Square test should pass");
}

void test_saxpy_cuda() {
    std::cout << "\n=== Test: SAXPY (CUDA) ===" << std::endl;

    if (!isBackendAvailable("cuda")) {
        std::cout << "SKIP: CUDA backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "cuda", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create CUDA device" << std::endl;
        return;
    }

    float alpha = 2.0f;
    std::vector<float> x(1024);
    std::vector<float> y(1024);
    for (size_t i = 0; i < 1024; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>(i + 1) * 0.1f;
    }

    TestKernel test;
    test.name = "SAXPY";
    test.source = saxpyKernelCuda;
    test.entryPoint = "saxpy_kernel";
    test.inputNames = {
        {"alpha", sizeof(float)},
        {"x", x.size() * sizeof(float)},
        {"y", y.size() * sizeof(float)}
    };
    test.outputNames = {{"result", 1024 * sizeof(float)}};
    test.initializeInputs = [&alpha, &x, &y](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], &alpha, sizeof(float));
        std::memcpy(inputPtrs[1], x.data(), x.size() * sizeof(float));
        std::memcpy(inputPtrs[2], y.data(), y.size() * sizeof(float));
    };
    test.validateOutputs = [&alpha, &x, &y](const std::vector<void*>& outputPtrs) {
        const float* resultPtr = static_cast<const float*>(outputPtrs[0]);
        for (size_t i = 0; i < 16; ++i) {
            float expected = alpha * x[i] + y[i];
            float actual = resultPtr[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "cuda", test, &err);
    assert(success && "CUDA SAXPY test should pass");
}

void test_square_hip() {
    std::cout << "\n=== Test: Square (HIP) ===" << std::endl;

    if (!isBackendAvailable("hip")) {
        std::cout << "SKIP: HIP backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "hip", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create HIP device" << std::endl;
        return;
    }

    std::vector<float> input(1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    TestKernel test;
    test.name = "Square";
    test.source = squareKernelHip;
    test.entryPoint = "square_kernel";
    test.inputNames = {{"input", input.size() * sizeof(float)}};
    test.outputNames = {{"output", input.size() * sizeof(float)}};
    test.initializeInputs = [&input](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], input.data(), input.size() * sizeof(float));
    };
    test.validateOutputs = [&input](const std::vector<void*>& outputPtrs) {
        const float* outputData = static_cast<const float*>(outputPtrs[0]);
        for (size_t i = 0; i < 16; ++i) {
            float expected = input[i] * input[i];
            float actual = outputData[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "hip", test, &err);
    assert(success && "HIP Square test should pass");
}

void test_saxpy_hip() {
    std::cout << "\n=== Test: SAXPY (HIP) ===" << std::endl;

    if (!isBackendAvailable("hip")) {
        std::cout << "SKIP: HIP backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "hip", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create HIP device" << std::endl;
        return;
    }

    float alpha = 2.0f;
    std::vector<float> x(1024);
    std::vector<float> y(1024);
    for (size_t i = 0; i < 1024; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>(i + 1) * 0.1f;
    }

    TestKernel test;
    test.name = "SAXPY";
    test.source = saxpyKernelHip;
    test.entryPoint = "saxpy_kernel";
    test.inputNames = {
        {"alpha", sizeof(float)},
        {"x", x.size() * sizeof(float)},
        {"y", y.size() * sizeof(float)}
    };
    test.outputNames = {{"result", 1024 * sizeof(float)}};
    test.initializeInputs = [&alpha, &x, &y](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], &alpha, sizeof(float));
        std::memcpy(inputPtrs[1], x.data(), x.size() * sizeof(float));
        std::memcpy(inputPtrs[2], y.data(), y.size() * sizeof(float));
    };
    test.validateOutputs = [&alpha, &x, &y](const std::vector<void*>& outputPtrs) {
        const float* resultPtr = static_cast<const float*>(outputPtrs[0]);
        for (size_t i = 0; i < 16; ++i) {
            float expected = alpha * x[i] + y[i];
            float actual = resultPtr[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "hip", test, &err);
    assert(success && "HIP SAXPY test should pass");
}

void test_param_buffer_overwrite_order_hip() {
    std::cout << "\n=== Test: Param Buffer Overwrite Ordering (HIP) ===" << std::endl;

    if (!isBackendAvailable("hip")) {
        std::cout << "SKIP: HIP backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "hip", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create HIP device" << std::endl;
        return;
    }

    const std::string hipSrc = R"HIP(
    #include <hip/hip_runtime.h>
    typedef unsigned int uint;
    typedef struct { uint n; float alpha; } Params;
    extern "C" __global__ void write_alpha(float* out, const Params* params) {
        uint i = (uint)(blockIdx.x * blockDim.x + threadIdx.x);
        if (i >= params->n) return;
        out[i] = params->alpha;
    }
    )HIP";

    std::string arch = device->getComputeCapability();
    if (auto colon = arch.find(':'); colon != std::string::npos) arch = arch.substr(0, colon);
    if (const char* overrideArch = std::getenv("GPU_HIP_ARCH")) {
        if (overrideArch[0] != '\0') arch = overrideArch;
    }

    std::vector<std::uint8_t> hsaco;
    std::string compileErr;
    if (!compileHipToHsaco(hipSrc, arch, &hsaco, &compileErr)) {
        throw std::runtime_error("Failed to compile HSACO via hipcc: " + compileErr);
    }

    ProgramSource ps;
    ps.kind = ProgramSource::Kind::Hsaco;
    ps.bytes = hsaco;
    ps.entryPoint = "write_alpha";
    auto program = device->createProgram(ps);
    auto q = device->createQueue();

    struct Params { std::uint32_t n; float alpha; };
    constexpr std::uint32_t n = 256;
    const size_t bytes = sizeof(float) * n;

    // Use separate param buffers for each kernel to avoid race condition.
    // Reusing dParams would cause a race: memcpy2 could overwrite dParams while kernel1
    // is still reading it (kernel launch is async, memcpy starts after launch, not completion).
    auto hpParams1 = device->createBuffer(sizeof(Params), MemoryKind::HostPinned);
    auto hpParams2 = device->createBuffer(sizeof(Params), MemoryKind::HostPinned);
    auto dParams1 = device->createBuffer(sizeof(Params), MemoryKind::Device);
    auto dParams2 = device->createBuffer(sizeof(Params), MemoryKind::Device);

    auto dOut1 = device->createBuffer(bytes, MemoryKind::Device);
    auto dOut2 = device->createBuffer(bytes, MemoryKind::Device);
    auto hpOut1 = device->createBuffer(bytes, MemoryKind::HostPinned);
    auto hpOut2 = device->createBuffer(bytes, MemoryKind::HostPinned);

    // Enqueue: params1=1.0 -> kernel(out1, params1), params2=2.0 -> kernel(out2, params2)
    {
        Params p{n, 1.0f};
        std::memcpy(hpParams1->mapPtr(MapMode::Write), &p, sizeof(Params));
        hpParams1->unmap();
        q->memcpy(*dParams1, *hpParams1, sizeof(Params));
    }
    {
        std::vector<Buffer*> args{dOut1.get(), dParams1.get()};
        q->submit(*program, args, /*grid*/ 1, 1, 1, /*block*/ n, 1, 1);
    }
    {
        Params p{n, 2.0f};
        std::memcpy(hpParams2->mapPtr(MapMode::Write), &p, sizeof(Params));
        hpParams2->unmap();
        q->memcpy(*dParams2, *hpParams2, sizeof(Params));
    }
    {
        std::vector<Buffer*> args{dOut2.get(), dParams2.get()};
        q->submit(*program, args, /*grid*/ 1, 1, 1, /*block*/ n, 1, 1);
    }
    q->memcpy(*hpOut1, *dOut1, bytes);
    q->memcpy(*hpOut2, *dOut2, bytes);
    q->synchronize();

    const float* out1 = static_cast<const float*>(hpOut1->mapPtr(MapMode::Read));
    const float* out2 = static_cast<const float*>(hpOut2->mapPtr(MapMode::Read));

    float maxErr1 = 0.0f;
    float maxErr2 = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        maxErr1 = std::max(maxErr1, std::abs(out1[i] - 1.0f));
        maxErr2 = std::max(maxErr2, std::abs(out2[i] - 2.0f));
    }
    hpOut1->unmap();
    hpOut2->unmap();

    std::cout << "maxErr(out1 vs 1.0)=" << maxErr1 << "\n";
    std::cout << "maxErr(out2 vs 2.0)=" << maxErr2 << "\n";
    assert(maxErr1 < 1e-6f && "Kernel 1 must see alpha=1.0");
    assert(maxErr2 < 1e-6f && "Kernel 2 must see alpha=2.0");
}

void test_square_metal() {
    std::cout << "\n=== Test: Square (Metal) ===" << std::endl;

    if (!isBackendAvailable("metal")) {
        std::cout << "SKIP: Metal backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "metal", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create Metal device" << std::endl;
        return;
    }

    std::vector<float> input(1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    TestKernel test;
    test.name = "Square";
    test.source = squareKernelMsl;
    test.entryPoint = "square_kernel";
    test.inputNames = {{"input", input.size() * sizeof(float)}};
    test.outputNames = {{"output", input.size() * sizeof(float)}};
    test.initializeInputs = [&input](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], input.data(), input.size() * sizeof(float));
    };
    test.validateOutputs = [&input](const std::vector<void*>& outputPtrs) {
        const float* outputData = static_cast<const float*>(outputPtrs[0]);
        for (size_t i = 0; i < 16; ++i) {
            float expected = input[i] * input[i];
            float actual = outputData[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "metal", test, &err);
    assert(success && "Metal Square test should pass");
}

void test_saxpy_metal() {
    std::cout << "\n=== Test: SAXPY (Metal) ===" << std::endl;

    if (!isBackendAvailable("metal")) {
        std::cout << "SKIP: Metal backend not available" << std::endl;
        return;
    }

    setenv("GPU_RUNTIME_BACKEND", "metal", 1);
    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: Could not create Metal device" << std::endl;
        return;
    }

    float alpha = 2.0f;
    std::vector<float> x(1024);
    std::vector<float> y(1024);
    for (size_t i = 0; i < 1024; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>(i + 1) * 0.1f;
    }

    TestKernel test;
    test.name = "SAXPY";
    test.source = saxpyKernelMsl;
    test.entryPoint = "saxpy_kernel";
    test.inputNames = {
        {"alpha", sizeof(float)},
        {"x", x.size() * sizeof(float)},
        {"y", y.size() * sizeof(float)}
    };
    test.outputNames = {{"result", 1024 * sizeof(float)}};
    test.initializeInputs = [&alpha, &x, &y](const std::vector<void*>& inputPtrs) {
        std::memcpy(inputPtrs[0], &alpha, sizeof(float));
        std::memcpy(inputPtrs[1], x.data(), x.size() * sizeof(float));
        std::memcpy(inputPtrs[2], y.data(), y.size() * sizeof(float));
    };
    test.validateOutputs = [&alpha, &x, &y](const std::vector<void*>& outputPtrs) {
        const float* resultPtr = static_cast<const float*>(outputPtrs[0]);
        for (size_t i = 0; i < 16; ++i) {
            float expected = alpha * x[i] + y[i];
            float actual = resultPtr[i];
            float tolerance = 0.0001f;
            if (std::abs(actual - expected) > tolerance) {
                std::cout << "FAIL: Element " << i << ": expected " << expected
                          << " but got " << actual << std::endl;
                throw std::runtime_error("Validation failed");
            }
        }
        std::cout << "Verified: First 16 elements correct" << std::endl;
    };

    std::string err;
    bool success = runKernelTest(*device, "metal", test, &err);
    assert(success && "Metal SAXPY test should pass");
}

void test_backend_availability() {
    std::cout << "\n=== Test: Backend Availability ===" << std::endl;

    std::vector<std::string> backends = {"opencl", "vulkan", "cuda", "hip", "metal"};

    for (const auto& backend : backends) {
        bool available = isBackendAvailable(backend);
        std::cout << backend << ": " << (available ? "AVAILABLE" : "NOT AVAILABLE") << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "GPU Runtime: End-to-End Kernel Execution Tests" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Test backend availability first
    test_backend_availability();

    // Run kernel tests
    test_square_opencl();
    test_saxpy_opencl();
    test_square_vulkan();
    test_square_cuda();
    test_saxpy_cuda();
    test_square_hip();
    test_saxpy_hip();
    test_param_buffer_overwrite_order_hip();
    test_square_metal();
    test_saxpy_metal();

    std::cout << "\n==================================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
