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

// GPU Runtime - CUDA Backend Implementation
// Full CUDA backend with Device, Queue, Buffer, Program, Event, Graph

#include <gpu_runtime.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_set>

namespace GpuRuntime {

namespace {

std::string cudaDriverErrorString(CUresult result) {
    const char* name = nullptr;
    const char* desc = nullptr;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &desc);
    std::ostringstream os;
    os << static_cast<int>(result);
    if (name && *name) {
        os << " " << name;
    }
    if (desc && *desc) {
        os << " " << desc;
    }
    return os.str();
}

std::string cudaRuntimeErrorString(cudaError_t result) {
    std::ostringstream os;
    os << static_cast<int>(result);
    const char* name = cudaGetErrorName(result);
    const char* desc = cudaGetErrorString(result);
    if (name && *name) {
        os << " " << name;
    }
    if (desc && *desc) {
        os << " " << desc;
    }
    return os.str();
}

struct CudaAllocationRegistry {
    std::mutex mutex;
    std::unordered_set<void*> hostPinned;
    std::unordered_set<void*> cudaFreed;
};

CudaAllocationRegistry& cudaAllocationRegistry() {
    static CudaAllocationRegistry registry;
    return registry;
}

} // namespace

// ============================================================================
// CudaEvent - Native CUDA event implementation
// ============================================================================

class CudaEvent : public Event {
public:
    CudaEvent() {
        (void)cudaGetDevice(&deviceId_);
        (void)cuCtxGetCurrent(&ctx_);
        if (cudaEventCreate(&event_, cudaEventDisableTiming) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA event");
        }
    }

    ~CudaEvent() override {
        restoreContext_();
        cudaEventDestroy(event_);
    }

    bool isComplete() const override {
        cudaError_t result = cudaEventQuery(event_);
        return result == cudaSuccess;
    }

    void wait() override {
        if (cudaEventSynchronize(event_) != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize CUDA event");
        }
    }

    void* nativeHandle() override {
        return reinterpret_cast<void*>(event_);
    }

    cudaEvent_t get() const { return event_; }

private:
    void restoreContext_() const {
        if (ctx_) {
            (void)cuCtxSetCurrent(ctx_);
        } else if (deviceId_ >= 0) {
            (void)cudaSetDevice(deviceId_);
        }
    }

    cudaEvent_t event_;
    int deviceId_ = -1;
    CUcontext ctx_ = nullptr;
};

// ============================================================================
// CudaQueue - CUDA stream (queue) implementation
// ============================================================================

class CudaQueue : public Queue {
public:
    CudaQueue(StreamPriority priority) {
        (void)cudaGetDevice(&deviceId_);
        (void)cuCtxGetCurrent(&ctx_);
        int priority_value = 0;
        int leastPriority = 0;
        int greatestPriority = 0;
        (void)cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        switch (priority) {
            case StreamPriority::Low:
                priority_value = leastPriority;
                break;
            case StreamPriority::High:
                priority_value = greatestPriority;
                break;
            case StreamPriority::Normal:
                // Prefer 0 if it's in range, otherwise fall back to leastPriority.
                if (greatestPriority <= 0 && 0 <= leastPriority) {
                    priority_value = 0;
                } else {
                    priority_value = leastPriority;
                }
                break;
        }

        if (cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, priority_value) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~CudaQueue() override {
        restoreContext_();
        cudaStreamDestroy(stream_);
    }

    void submit(Program& program, const std::vector<Buffer*>& args,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z) override {
        restoreContext_();
        // Get CUDA function from program
        CUfunction kernel = static_cast<CUfunction>(program.nativeHandle());

        // Prepare kernel arguments
        // cuLaunchKernel expects `void**` where each element points to the argument value.
        // For pointer arguments, the argument value is a CUdeviceptr.
        std::vector<CUdeviceptr> devicePtrs;
        devicePtrs.reserve(args.size());
        for (auto* buf : args) {
            devicePtrs.push_back(reinterpret_cast<CUdeviceptr>(buf->getDevicePointer()));
        }
        std::vector<void*> kernelArgs;
        kernelArgs.reserve(devicePtrs.size());
        for (auto& p : devicePtrs) {
            kernelArgs.push_back(&p);
        }

        // Launch kernel
        CUresult launchResult = cuLaunchKernel(kernel,
                                               grid_x, grid_y, grid_z,
                                               block_x, block_y, block_z,
                                               /*sharedMemBytes*/ 0,
                                               static_cast<CUstream>(stream_),
                                               kernelArgs.data(),
                                               /*extra*/ nullptr);
        if (launchResult != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to launch CUDA kernel: "
                                     + cudaDriverErrorString(launchResult));
        }
    }

    void memcpy(Buffer& dst, const Buffer& src, size_t size) override {
        restoreContext_();
        // Use cudaMemcpyDefault so mapped pinned/managed pointers are handled correctly.
        cudaError_t result = cudaMemcpyAsync(dst.getDevicePointer(), const_cast<Buffer&>(src).getDevicePointer(), size,
                                             cudaMemcpyDefault, stream_);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to enqueue CUDA memcpy: "
                                     + cudaRuntimeErrorString(result));
        }
    }

    void synchronize() override {
        restoreContext_();
        cudaError_t result = cudaStreamSynchronize(stream_);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize CUDA stream: "
                                     + cudaRuntimeErrorString(result));
        }
    }

    void* nativeHandle() override {
        return stream_;
    }

    // Event recording
    void record(Event& event) override {
        restoreContext_();
        CudaEvent* cudaEvent = static_cast<CudaEvent*>(&event);
        if (cudaEventRecord(cudaEvent->get(), stream_) != cudaSuccess) {
            throw std::runtime_error("Failed to record CUDA event");
        }
    }

    // Wait for event
    void wait(Event& event) override {
        restoreContext_();
        CudaEvent* cudaEvent = static_cast<CudaEvent*>(&event);
        if (cudaStreamWaitEvent(stream_, cudaEvent->get(), 0) != cudaSuccess) {
            throw std::runtime_error("Failed to wait on CUDA event");
        }
    }

    // Wait with timeout
    bool waitFor(Event& event, uint64_t timeout_us) override {
        restoreContext_();
        CudaEvent* cudaEvent = static_cast<CudaEvent*>(&event);
        cudaError_t result = cudaEventQuery(cudaEvent->get());
        if (result == cudaSuccess) return true;
        if (result == cudaErrorNotReady) {
            // Wait with timeout (simplified: just poll)
            auto start = std::chrono::steady_clock::now();
            while (cudaEventQuery(cudaEvent->get()) == cudaErrorNotReady) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
                if (elapsed.count() >= timeout_us) return false;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            return true;
        }
        return false;
    }

    // Multi-queue synchronization
    void wait(Event& event, Queue& other) override {
        restoreContext_();
        CudaEvent* cudaEvent = static_cast<CudaEvent*>(&event);
        (void)other;
        if (cudaStreamWaitEvent(stream_, cudaEvent->get(), 0) != cudaSuccess) {
            throw std::runtime_error("Failed to wait on CUDA event from other queue");
        }
    }

private:
    void restoreContext_() const {
        if (ctx_) {
            (void)cuCtxSetCurrent(ctx_);
        } else if (deviceId_ >= 0) {
            (void)cudaSetDevice(deviceId_);
        }
    }

    cudaStream_t stream_;
    int deviceId_ = -1;
    CUcontext ctx_ = nullptr;
};

// ============================================================================
// CudaBuffer - CUDA device memory buffer
// ============================================================================

class CudaBuffer : public Buffer {
public:
    CudaBuffer(size_t size, MemoryKind kind) : size_(size), kind_(kind) {
        (void)cudaGetDevice(&deviceId_);
        (void)cuCtxGetCurrent(&ctx_);
        if (size == 0) {
            throw std::runtime_error(
                "CudaBuffer: zero-byte allocation requested (kind="
                + std::string((kind == MemoryKind::Device) ? "Device"
                    : (kind == MemoryKind::HostPinned) ? "HostPinned" : "Managed")
                + ")");
        }
        cudaError_t result;
        void* ptr = nullptr;
        void* devPtr = nullptr;

        switch (kind) {
            case MemoryKind::Device:
                result = cudaMalloc(&ptr, size);
                break;
            case MemoryKind::HostPinned:
                // Allocate pinned host memory mapped into the device address space, so
                // `getDevicePointer()` is valid for kernel launches.
                result = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
                break;
            case MemoryKind::Managed:
                result = cudaMallocManaged(&ptr, size);
                break;
        }

        if (result != cudaSuccess || !ptr) {
            const char* kindStr = (kind == MemoryKind::Device) ? "Device"
                : (kind == MemoryKind::HostPinned) ? "HostPinned" : "Managed";
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            throw std::runtime_error(
                std::string("Failed to allocate CUDA buffer: ")
                + cudaGetErrorString(result)
                + " (kind=" + kindStr
                + ", requested=" + std::to_string(size) + " bytes"
                + ", free=" + std::to_string(freeMem)
                + ", total=" + std::to_string(totalMem) + ")");
        }

        if (kind == MemoryKind::HostPinned) {
            if (cudaHostGetDevicePointer(&devPtr, ptr, 0) != cudaSuccess) {
                cudaFreeHost(ptr);
                throw std::runtime_error("Failed to get device pointer for mapped pinned buffer");
            }
            hostPtr_ = ptr;
            devicePtr_ = devPtr;
        } else if (kind == MemoryKind::Managed) {
            hostPtr_ = ptr;
            devicePtr_ = ptr;
        } else {
            hostPtr_ = nullptr;
            devicePtr_ = ptr;
        }

        auto& registry = cudaAllocationRegistry();
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (kind == MemoryKind::HostPinned) {
                registry.hostPinned.insert(hostPtr_);
            } else {
                registry.cudaFreed.insert(devicePtr_);
            }
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&&) = delete;
    CudaBuffer& operator=(CudaBuffer&&) = delete;

    ~CudaBuffer() override {
        restoreContext_();
        void* hostToFree = nullptr;
        void* deviceToFree = nullptr;
        auto& registry = cudaAllocationRegistry();
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (hostPtr_ && registry.hostPinned.erase(hostPtr_) != 0) {
                hostToFree = hostPtr_;
            } else if (devicePtr_ && registry.cudaFreed.erase(devicePtr_) != 0) {
                deviceToFree = devicePtr_;
            }
        }

        // Trust the allocation registry over kind_: if an object's metadata is
        // corrupted by an upstream overwrite, calling the wrong CUDA free API can
        // abort the MPI rank during normal teardown. Skipping an unknown pointer is
        // preferable to crashing after a completed run; the registry still frees
        // all allocations it created.
        //
        // Synchronize before freeing pinned/device memory: if a cudaMemcpyAsync
        // or kernel launch still holds a reference to this buffer, freeing it
        // would cause the DMA/kernel to read/write released memory and corrupt
        // downstream device state (debrief 2026-04-23, cell-105 divergence).
        if (hostToFree) {
            if (std::getenv("GPU_RUNTIME_LEAK_HOSTPINNED_ON_EXIT") == nullptr) {
                (void)cudaDeviceSynchronize();
                (void)cudaFreeHost(hostToFree);
            }
        } else if (deviceToFree) {
            (void)cudaDeviceSynchronize();
            (void)cudaFree(deviceToFree);
        } else if (std::getenv("GPU_RUNTIME_DEBUG_BUFFER_FREE") != nullptr) {
            std::cerr << "CudaBuffer: skipped unregistered pointer during teardown"
                      << " kind=" << static_cast<int>(kind_)
                      << " hostPtr=" << hostPtr_
                      << " devicePtr=" << devicePtr_
                      << " size=" << size_ << std::endl;
        }
        hostPtr_ = nullptr;
        devicePtr_ = nullptr;
    }

    size_t sizeBytes() const override { return size_; }
    MemoryKind memoryKind() const override { return kind_; }

    void* getDevicePointer() override { return devicePtr_; }

    void* getHostPointer() override {
        if (kind_ == MemoryKind::HostPinned) {
            return hostPtr_;
        }
        if (kind_ == MemoryKind::Managed) {
            return hostPtr_;
        }
        // For device memory, need to map
        if (!hostPtr_) {
            return nullptr;
        }
        return hostPtr_;
    }

    void map(MapMode mode) override {
        // HostPinned/Managed are already host-accessible, but any pending async
        // H->D memcpy issued from this buffer must complete before the CPU
        // writes fresh data into it, otherwise the DMA engine reads a mix of
        // old and new bytes. Mirror the semantic of mapPtr() as a fence for
        // the current device: sync to drain pending work before the caller
        // gets the host pointer.
        //
        // Without this, kernels that reuse persistent host-pinned staging
        // buffers across repeated compute() calls race with their own prior
        // DMA ops (debrief 2026-04-23, cell-105 divergence on procBoundary).
        (void)mode;
        if (kind_ == MemoryKind::HostPinned || kind_ == MemoryKind::Managed) {
            restoreContext_();
            (void)cudaDeviceSynchronize();
        }
        // Device memory mapping is not currently supported.
    }

    void unmap() override {
        // no-op for HostPinned/Managed.
    }

    void* nativeHandle() const override {
        return devicePtr_;
    }

private:
    void restoreContext_() const {
        if (ctx_) {
            (void)cuCtxSetCurrent(ctx_);
        } else if (deviceId_ >= 0) {
            (void)cudaSetDevice(deviceId_);
        }
    }

    size_t size_;
    MemoryKind kind_;
    void* devicePtr_;
    void* hostPtr_;
    int deviceId_ = -1;
    CUcontext ctx_ = nullptr;
};

// ============================================================================
// CudaProgram - Compiled CUDA kernel program
// ============================================================================

class CudaProgram : public Program {
public:
    CudaProgram(const ProgramSource& source) {
        (void)cudaGetDevice(&deviceId_);
        // Ensure a current driver context exists (created by CudaDevice).
        CUcontext ctx = nullptr;
        if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
            throw std::runtime_error("No current CUDA driver context (create a Device first)");
        }
        ctx_ = ctx;

        // We support two input formats:
        // - PTX (precompiled)
        // - CUDA C++ source (JIT via NVRTC to PTX)
        if (source.kind == ProgramSource::Kind::CudaCpp) {
            std::string cudaSrc(reinterpret_cast<const char*>(source.bytes.data()), source.bytes.size());
            if (cudaSrc.empty() || cudaSrc.back() != '\0') {
                cudaSrc.push_back('\0');
            }

            // Target architecture from the active context's device.
            CUdevice dev = 0;
            int major = 0;
            int minor = 0;
            if (cuCtxGetDevice(&dev) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) {
                throw std::runtime_error("Failed to query CUDA device compute capability for NVRTC");
            }

            std::string cudaRoot = "/usr/local/cuda";
            if (const char* env = std::getenv("CUDA_HOME")) cudaRoot = env;
            else if (const char* env2 = std::getenv("CUDA_PATH")) cudaRoot = env2;

            // Cap NVRTC architecture at compute_90 for forward compatibility.
            // sm_120 (Blackwell) PTX compiled for compute_90 runs correctly
            // via PTX forward compatibility. Direct compute_120 may crash
            // on some NVRTC versions.
            int nvrtcMajor = major, nvrtcMinor = minor;
            if (major > 9 || (major == 9 && minor > 0)) {
                nvrtcMajor = 9; nvrtcMinor = 0;
            }
            const std::string archOpt = "--gpu-architecture=compute_" + std::to_string(nvrtcMajor) + std::to_string(nvrtcMinor);
            const std::string includeOpt = "-I" + cudaRoot + "/include";
            std::vector<std::string> optsStorage = {
                "--std=c++17",
                archOpt,
                includeOpt
            };
            std::vector<const char*> opts;
            opts.reserve(optsStorage.size());
            for (const auto& s : optsStorage) {
                opts.push_back(s.c_str());
            }

            nvrtcProgram prog;
            nvrtcResult createRes = nvrtcCreateProgram(&prog, cudaSrc.c_str(), nullptr, 0, nullptr, nullptr);
            if (createRes != NVRTC_SUCCESS) {
                throw std::runtime_error(std::string("NVRTC: failed to create program: ") + nvrtcGetErrorString(createRes));
            }

            nvrtcResult compRes = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
            if (compRes != NVRTC_SUCCESS) {
                size_t logSize = 0;
                nvrtcGetProgramLogSize(prog, &logSize);
                std::string log(logSize, '\0');
                if (logSize > 0) {
                    nvrtcGetProgramLog(prog, log.data());
                }
                nvrtcDestroyProgram(&prog);
                throw std::runtime_error(std::string("NVRTC: failed to compile CUDA source (") +
                                         nvrtcGetErrorString(compRes) + "): " + log);
            }

            size_t ptxSize = 0;
            nvrtcResult ptxSizeRes = nvrtcGetPTXSize(prog, &ptxSize);
            if (ptxSizeRes != NVRTC_SUCCESS || ptxSize == 0) {
                nvrtcDestroyProgram(&prog);
                throw std::runtime_error("NVRTC: failed to get PTX size");
            }

            std::string ptx(ptxSize, '\0');
            nvrtcResult ptxRes = nvrtcGetPTX(prog, ptx.data());
            nvrtcDestroyProgram(&prog);
            if (ptxRes != NVRTC_SUCCESS) {
                throw std::runtime_error("NVRTC: failed to get PTX");
            }

            compiledBytes_.assign(reinterpret_cast<const std::uint8_t*>(ptx.data()),
                                  reinterpret_cast<const std::uint8_t*>(ptx.data()) + ptx.size());
            compiledFormat_ = "ptx";
        } else if (source.kind == ProgramSource::Kind::Ptx) {
            compiledBytes_ = source.bytes;
            compiledFormat_ = "ptx";
        } else if (source.kind == ProgramSource::Kind::Cubin) {
            compiledBytes_ = source.bytes;
            compiledFormat_ = "cubin";
        } else {
            throw std::runtime_error("CUDA backend supports only PTX, CUBIN, or CUDA C++ source");
        }

        if (compiledBytes_.empty()) {
            throw std::runtime_error("CUDA backend received an empty " + compiledFormat_ + " image");
        }

        // Load CUDA module from PTX.
        // Keep this conservative: avoid JIT option plumbing until we have a validated policy.
        CUresult loadRes = CUDA_ERROR_INVALID_IMAGE;
        if (compiledFormat_ == "ptx") {
            // `cuModuleLoadData` expects PTX as a null-terminated C-string.
            std::string ptxText(reinterpret_cast<const char*>(compiledBytes_.data()), compiledBytes_.size());
            if (ptxText.empty() || ptxText.back() != '\0') {
                ptxText.push_back('\0');
            }
            loadRes = cuModuleLoadData(&module_, ptxText.c_str());
        } else {
            loadRes = cuModuleLoadData(&module_, compiledBytes_.data());
        }
        if (loadRes != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to load CUDA module from " + compiledFormat_ + " (cuModuleLoadData: "
                + cudaDriverErrorString(loadRes) + ")");
        }

        // Get kernel function
        if (cuModuleGetFunction(&kernel_, module_, source.entryPoint.c_str()) != CUDA_SUCCESS) {
            cuModuleUnload(module_);
            throw std::runtime_error("Failed to get CUDA kernel function");
        }
    }

    ~CudaProgram() override {
        restoreContext_();
        if (module_) {
            cuModuleUnload(module_);
        }
    }

    void* nativeHandle() override {
        return kernel_;
    }

    // Get compiled PTX bytes for disk caching
    std::vector<std::uint8_t> getCompiledBytes() const override {
        return compiledBytes_;
    }

    // Get compiled format identifier
    std::string getCompiledFormat() const override {
        return compiledFormat_;
    }

private:
    void restoreContext_() const {
        if (ctx_) {
            (void)cuCtxSetCurrent(ctx_);
        } else if (deviceId_ >= 0) {
            (void)cudaSetDevice(deviceId_);
        }
    }

    CUmodule module_ = nullptr;
    CUfunction kernel_ = nullptr;
    std::vector<std::uint8_t> compiledBytes_;  // Store original PTX/CUBIN bytes.
    std::string compiledFormat_ = "ptx";
    int deviceId_ = -1;
    CUcontext ctx_ = nullptr;
};

// ============================================================================
// CudaGraph - CUDA graph capture and execution
// ============================================================================

class CudaGraph : public Graph {
public:
    CudaGraph() : graph_(nullptr), exec_(nullptr) {}

    ~CudaGraph() override {
        if (exec_) {
            cuGraphExecDestroy(exec_);
        }
        if (graph_) {
            cuGraphDestroy(graph_);
        }
    }

    void launch(Queue& queue) override {
        if (!exec_) {
            throw std::runtime_error("Cannot launch uninstantiated CUDA graph");
        }

        CUstream stream = static_cast<CUstream>(queue.nativeHandle());
        if (cuGraphLaunch(exec_, stream) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to launch CUDA graph");
        }
    }

    void update() override {
        // Update graph topology after parameter mutations
        // For now, just re-instantiate
        if (graph_ && exec_) {
            cuGraphExecDestroy(exec_);
            exec_ = nullptr;

            if (cuGraphInstantiate(&exec_, graph_, 0) != CUDA_SUCCESS) {
                throw std::runtime_error("Failed to re-instantiate CUDA graph");
            }
        }
    }

    void beginCapture(Queue& queue) override {
        cudaStream_t stream = static_cast<cudaStream_t>(queue.nativeHandle());
        if (exec_) {
            cuGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_) {
            cuGraphDestroy(graph_);
            graph_ = nullptr;
        }

        if (cuStreamBeginCapture(static_cast<CUstream>(stream), CU_STREAM_CAPTURE_MODE_GLOBAL) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to begin CUDA graph capture");
        }
    }

    void endCapture(Queue& queue) override {
        cudaStream_t stream = static_cast<cudaStream_t>(queue.nativeHandle());
        if (cuStreamEndCapture(static_cast<CUstream>(stream), &graph_) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to end CUDA graph capture");
        }

        if (cuGraphInstantiate(&exec_, graph_, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to instantiate CUDA graph");
        }
    }

    CUgraph get() const { return graph_; }

private:
    CUgraph graph_;
    CUgraphExec exec_;
};

// ============================================================================
// CudaDevice - CUDA device implementation
// ============================================================================

class CudaDevice : public Device {
public:
    CudaDevice(int device_id) {
        // Initialize driver API once per process.
        static bool driverInit = []() {
            return (cuInit(0) == CUDA_SUCCESS);
        }();
        if (!driverInit) {
            throw std::runtime_error("Failed to initialize CUDA driver API");
        }

        if (cudaSetDevice(device_id) != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device");
        }

        if (cudaGetDeviceProperties(&props_, device_id) != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device properties");
        }

        device_id_ = device_id;

        // Ensure a current driver context is active for this device (for PTX module loading).
        if (cuDeviceGet(&cuDevice_, device_id) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device handle");
        }
        if (cuDevicePrimaryCtxRetain(&cuCtx_, cuDevice_) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to retain CUDA primary context");
        }
        if (cuCtxSetCurrent(cuCtx_) != CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(cuDevice_);
            cuCtx_ = nullptr;
            throw std::runtime_error("Failed to set current CUDA context");
        }
    }

    ~CudaDevice() override {
        if (cuCtx_) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(cuDevice_);
            cuCtx_ = nullptr;
        }
    }

    std::unique_ptr<Queue> createQueue(StreamPriority priority) override {
        return std::make_unique<CudaQueue>(priority);
    }

    std::unique_ptr<Buffer> createBuffer(size_t size, MemoryKind kind) override {
        return std::make_unique<CudaBuffer>(size, kind);
    }

    std::unique_ptr<Program> createProgram(const ProgramSource& source) override {
        if (cuCtxSetCurrent(cuCtx_) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to set CUDA context current for program creation");
        }
        return std::make_unique<CudaProgram>(source);
    }

    // Graph creation
    std::unique_ptr<Graph> createGraph() override {
        return std::make_unique<CudaGraph>();
    }

    // Event creation
    std::unique_ptr<Event> createEvent() override {
        return std::make_unique<CudaEvent>();
    }

    // Device metadata
    std::string getName() const override {
        return props_.name;
    }

    size_t getTotalMemory() const override {
        return props_.totalGlobalMem;
    }

    std::string getComputeCapability() const override {
        int major = props_.major;
        int minor = props_.minor;
        return std::to_string(major) + "." + std::to_string(minor);
    }

    bool supportsFeature(const std::string& feature) const override {
        if (feature == "float64" || feature == "fp64") return true;
        if (feature == "float16" || feature == "fp16") return true;
        if (feature == "bfloat16" || feature == "bf16") return props_.major >= 8;
        if (feature == "graphs") return props_.major >= 5;  // Pascal+
        if (feature == "managedMemory") return props_.managedMemory;
        if (feature == "sharedMem") return true;
        if (feature == "streams") return true;
        if (feature == "events") return true;
        if (feature == "tensor_cores") return props_.major >= 7;  // Volta+
        if (feature == "memory_pools") return true;
        if (feature == "nurbs.rational.evaluate") return true;
        if (feature == "nurbs.rational.derivatives") return true;
        if (feature == "nurbs.rational.project") return true;
        return false;
    }

    // Extended capability queries
    DeviceVersion getVersion() const override {
        return DeviceVersion{props_.major, props_.minor};
    }

    bool hasExtension(const std::string& extension) const override {
        // CUDA doesn't have extensions like Vulkan
        // Map feature names to compute capability checks
        if (extension == "cuda") return true;
        if (extension == "cudaComputeCapability") return true;
        return false;
    }

    uint32_t getSubgroupSize() const override {
        // Warp size is 32 for all CUDA devices
        return 32;
    }

    bool supportsAtomicOp(AtomicOpType op, DataType type) const override {
        // Check compute capability for atomic support
        int cc = props_.major * 10 + props_.minor;
        (void)op;  // Most atomic operations have the same requirements

        switch (type) {
            case DataType::Float64:
                return cc >= 60;  // Atomic double requires sm_60+
            case DataType::Float32:
                return cc >= 20;  // Atomic float requires sm_20+
            case DataType::Float16:
            case DataType::BFloat16:
                return cc >= 70;  // Atomic half requires sm_70+
            case DataType::Int64:
            case DataType::UInt64:
                return cc >= 20;  // 64-bit atomics require sm_20+
            case DataType::Int32:
            case DataType::UInt32:
            case DataType::Int16:
            case DataType::UInt16:
            case DataType::Int8:
            case DataType::UInt8:
                return true;  // 32-bit and smaller atomics always supported
        }
        return false;
    }

    size_t getMaxSharedMemoryPerWorkgroup() const override {
        return props_.sharedMemPerBlock;
    }

    uint32_t getMaxWorkgroupSize() const override {
        return props_.maxThreadsPerBlock;
    }

    std::vector<std::string> getSupportedExtensions() const override {
        // CUDA doesn't use extensions like Vulkan
        return {"cuda"};
    }

    bool queryMemoryInfo(size_t* freeBytes, size_t* totalBytes) const override {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
        if (freeBytes) *freeBytes = f;
        if (totalBytes) *totalBytes = t;
        return true;
    }

private:
    int device_id_;
    cudaDeviceProp props_;
    CUdevice cuDevice_ = 0;
    CUcontext cuCtx_ = nullptr;
};

// Factory function
std::unique_ptr<Device> CudaDevice_create(int device_id) {
    return std::make_unique<CudaDevice>(device_id);
}

} // namespace GpuRuntime
