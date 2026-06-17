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

// GPU Runtime - HIP Backend Implementation
// Full HIP backend with Device, Queue, Buffer, Program, Event, Graph

#include <gpu_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include "program_cache.h"

namespace GpuRuntime {

namespace {

static std::string shellQuote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') out.append("'\\''");
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

static std::string normalizeHipArch(std::string arch) {
    const std::size_t pos = arch.find(':');
    if (pos != std::string::npos) arch.resize(pos);
    return arch;
}

static std::string normalizeHipSource(const std::vector<std::uint8_t>& hipCpp) {
    std::string src(
        reinterpret_cast<const char*>(hipCpp.data()),
        hipCpp.size());

    // Many call sites provide raw HIP kernel bodies and rely on the JIT path to
    // make HIP builtins like blockIdx/threadIdx/atomicAdd visible.
    if (src.find("#include <hip/hip_runtime.h>") == std::string::npos) {
        src.insert(0, "#include <hip/hip_runtime.h>\n");
    }

    return src;
}

static std::vector<std::uint8_t> readFileBytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path.string());
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes;
    if (n <= 0) return bytes;
    bytes.resize(static_cast<std::size_t>(n));
    in.read(reinterpret_cast<char*>(bytes.data()), n);
    if (!in) throw std::runtime_error("Failed to read file: " + path.string());
    return bytes;
}

static void writeFileBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to write file: " + path.string());
    if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("Failed to write file: " + path.string());
}

static std::vector<std::uint8_t> compileHipCppToHsaco(
    const std::vector<std::uint8_t>& hipCpp,
    const std::string& archRaw) {
    const char* hipccEnv = std::getenv("HIPCC");
    const std::string hipccExe = hipccEnv ? std::string(hipccEnv) : "hipcc";
    const std::string arch = normalizeHipArch(archRaw);

    std::string tmpTemplate = (std::filesystem::temp_directory_path() / "gpu_runtime_hipcc_XXXXXX").string();
    std::vector<char> tmpBuf(tmpTemplate.begin(), tmpTemplate.end());
    tmpBuf.push_back('\0');
    char* dirC = ::mkdtemp(tmpBuf.data());
    if (!dirC) throw std::runtime_error("Failed to create temp dir for hipcc");
    const std::filesystem::path tmpDir(dirC);

    const std::filesystem::path srcPath = tmpDir / "kernel.hip";
    const std::filesystem::path outPath = tmpDir / "kernel.hsaco";
    const std::filesystem::path logPath = tmpDir / "hipcc.log";

    const std::string normalizedSource = normalizeHipSource(hipCpp);
    writeFileBytes(
        srcPath,
        std::vector<std::uint8_t>(normalizedSource.begin(), normalizedSource.end()));

    std::ostringstream cmd;
    cmd << shellQuote(hipccExe)
        << " --genco"
        << " --offload-arch=" << arch
        << " " << shellQuote(srcPath.string())
        << " -o " << shellQuote(outPath.string())
        << " 2> " << shellQuote(logPath.string());

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::string log;
        try {
            auto logBytes = readFileBytes(logPath);
            log.assign(logBytes.begin(), logBytes.end());
        } catch (...) {
            log = "<failed to read hipcc log>";
        }
        const char* keepFailedEnv = std::getenv("GPU_RUNTIME_KEEP_FAILED_HIPCC");
        if (!(keepFailedEnv && keepFailedEnv[0] != '\0' && std::string(keepFailedEnv) != "0")) {
            std::error_code ec;
            std::filesystem::remove_all(tmpDir, ec);
        } else {
            log += "\n[kept failed hipcc temp dir] " + tmpDir.string() + "\n";
        }
        throw std::runtime_error("hipcc failed to compile HIP source to HSACO (rc=" + std::to_string(rc) + "):\n" + log);
    }

    auto hsaco = readFileBytes(outPath);
    std::error_code ec;
    std::filesystem::remove_all(tmpDir, ec);
    return hsaco;
}

} // namespace

// ============================================================================
// HipEvent - Native HIP event implementation
// ============================================================================

class HipEvent : public Event {
public:
    HipEvent() {
        if (hipEventCreateWithFlags(&event_, hipEventDisableTiming) != hipSuccess) {
            throw std::runtime_error("Failed to create HIP event");
        }
    }

    ~HipEvent() override {
        hipEventDestroy(event_);
    }

    bool isComplete() const override {
        hipError_t result = hipEventQuery(event_);
        return result == hipSuccess;
    }

    void wait() override {
        if (hipEventSynchronize(event_) != hipSuccess) {
            throw std::runtime_error("Failed to synchronize HIP event");
        }
    }

    void* nativeHandle() override {
        return &event_;
    }

    hipEvent_t get() const { return event_; }

private:
    hipEvent_t event_;
};

// ============================================================================
// HipQueue - HIP stream (queue) implementation
// ============================================================================

class HipQueue : public Queue {
public:
    HipQueue(StreamPriority priority) {
        int leastPriority = 0;
        int greatestPriority = 0;
        (void)hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

        int requested = 0;
        switch (priority) {
            case StreamPriority::High: requested = greatestPriority; break;
            case StreamPriority::Low: requested = leastPriority; break;
            case StreamPriority::Normal: requested = 0; break;
        }
        // Clamp to [greatestPriority, leastPriority] (HIP follows CUDA: smaller value = higher priority).
        if (requested < greatestPriority) requested = greatestPriority;
        if (requested > leastPriority) requested = leastPriority;

        if (hipStreamCreateWithPriority(&stream_, hipStreamDefault, requested) != hipSuccess) {
            throw std::runtime_error("Failed to create HIP stream");
        }
    }

    ~HipQueue() override {
        hipStreamDestroy(stream_);
    }

    void submit(Program& program, const std::vector<Buffer*>& args,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z) override {
        // Get HIP function from program
        hipFunction_t kernel = static_cast<hipFunction_t>(program.nativeHandle());

        // hipModuleLaunchKernel expects an array of pointers-to-arguments.
        std::vector<void*> argValues;
        argValues.reserve(args.size());
        for (auto* buf : args) {
            argValues.push_back(buf->nativeHandle());
        }
        std::vector<void*> kernelParams;
        kernelParams.reserve(argValues.size());
        for (auto& v : argValues) {
            kernelParams.push_back(&v);
        }

        if (hipModuleLaunchKernel(
                kernel,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                0,
                stream_,
                kernelParams.data(),
                nullptr) != hipSuccess) {
            throw std::runtime_error("Failed to launch HIP kernel");
        }
    }

    void memcpy(Buffer& dst, const Buffer& src, size_t size) override {
        // Use hipMemcpyDefault so host-pinned/managed pointers are handled correctly.
        if (hipMemcpyAsync(dst.getDevicePointer(), const_cast<Buffer&>(src).getDevicePointer(), size,
                           hipMemcpyDefault, stream_) != hipSuccess) {
            throw std::runtime_error("Failed to enqueue HIP memcpy");
        }
    }

    void synchronize() override {
        if (hipStreamSynchronize(stream_) != hipSuccess) {
            throw std::runtime_error("Failed to synchronize HIP stream");
        }
    }

    void* nativeHandle() override {
        return stream_;
    }

    // Event recording
    void record(Event& event) override {
        HipEvent* hipEvent = static_cast<HipEvent*>(&event);
        if (hipEventRecord(hipEvent->get(), stream_) != hipSuccess) {
            throw std::runtime_error("Failed to record HIP event");
        }
    }

    // Wait for event
    void wait(Event& event) override {
        HipEvent* hipEvent = static_cast<HipEvent*>(&event);
        if (hipStreamWaitEvent(stream_, hipEvent->get(), 0) != hipSuccess) {
            throw std::runtime_error("Failed to wait on HIP event");
        }
    }

    // Wait with timeout
    bool waitFor(Event& event, uint64_t timeout_us) override {
        HipEvent* hipEvent = static_cast<HipEvent*>(&event);
        hipError_t result = hipEventQuery(hipEvent->get());
        if (result == hipSuccess) return true;
        if (result == hipErrorNotReady) {
            // Wait with timeout (simplified: just poll)
            auto start = std::chrono::steady_clock::now();
            while (hipEventQuery(hipEvent->get()) == hipErrorNotReady) {
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
        HipEvent* hipEvent = static_cast<HipEvent*>(&event);
        (void)other;
        if (hipStreamWaitEvent(stream_, hipEvent->get(), 0) != hipSuccess) {
            throw std::runtime_error("Failed to wait on HIP event from other queue");
        }
    }

private:
    hipStream_t stream_;
};

// ============================================================================
// HipBuffer - HIP device memory buffer
// ============================================================================

class HipBuffer : public Buffer {
public:
    HipBuffer(size_t size, MemoryKind kind) : size_(size), kind_(kind) {
        if (size == 0) {
            throw std::runtime_error(
                "HipBuffer: zero-byte allocation requested (kind="
                + std::string((kind == MemoryKind::Device) ? "Device"
                    : (kind == MemoryKind::HostPinned) ? "HostPinned" : "Managed")
                + ")");
        }
        hipError_t result;
        void* ptr = nullptr;
        void* devPtr = nullptr;

        switch (kind) {
            case MemoryKind::Device:
                result = hipMalloc(&ptr, size);
                break;
            case MemoryKind::HostPinned:
                // Allocate pinned host memory mapped into the device address space so
                // `getDevicePointer()` is valid for kernel launches.
                result = hipHostMalloc(&ptr, size, hipHostMallocMapped);
                break;
            case MemoryKind::Managed:
                result = hipMallocManaged(&ptr, size);
                break;
        }

        if (result != hipSuccess || !ptr) {
            const char* kindStr = (kind == MemoryKind::Device) ? "Device"
                : (kind == MemoryKind::HostPinned) ? "HostPinned" : "Managed";
            size_t freeMem = 0, totalMem = 0;
            hipMemGetInfo(&freeMem, &totalMem);
            throw std::runtime_error(
                std::string("Failed to allocate HIP buffer: ")
                + hipGetErrorString(result)
                + " (kind=" + kindStr
                + ", requested=" + std::to_string(size) + " bytes"
                + ", free=" + std::to_string(freeMem)
                + ", total=" + std::to_string(totalMem) + ")");
        }

        if (kind == MemoryKind::HostPinned) {
            if (hipHostGetDevicePointer(&devPtr, ptr, 0) != hipSuccess) {
                hipHostFree(ptr);
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
    }

    ~HipBuffer() override {
        if (kind_ == MemoryKind::HostPinned) {
            hipHostFree(hostPtr_);
        } else {
            hipFree(devicePtr_);
        }
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
        (void)mode;
        // HostPinned/Managed are already host-accessible. Device memory mapping is not currently supported.
    }

    void unmap() override {
        // no-op for HostPinned/Managed.
    }

    void* nativeHandle() const override {
        return devicePtr_;
    }

private:
    size_t size_;
    MemoryKind kind_;
    void* devicePtr_;
    void* hostPtr_;
};

// ============================================================================
// HipProgram - Compiled HIP kernel program
// ============================================================================

class HipProgram : public Program {
public:
    HipProgram(const ProgramSource& source, const std::string& arch) {
        if (source.kind == ProgramSource::Kind::Hsaco) {
            hsacoBytes_ = source.bytes;
        } else if (source.kind == ProgramSource::Kind::HipCpp) {
            const std::string archNorm = normalizeHipArch(arch);
            const std::string hipSource = normalizeHipSource(source.bytes);

            // Cache key includes target arch + entrypoint to avoid cross-device collisions.
            const std::string cacheKeySource =
                "arch=" + archNorm + "\n" +
                "entry=" + source.entryPoint + "\n" +
                hipSource;

            // JIT compile HIP C++ to HSACO via hipcc, but prefer the persistent disk cache.
            if (!tryLoadFromDiskCache("hip", source.entryPoint, cacheKeySource, hsacoBytes_)) {
                hsacoBytes_ = compileHipCppToHsaco(source.bytes, archNorm);
                (void)saveToDiskCache("hip", source.entryPoint, cacheKeySource, hsacoBytes_);
            }
        } else {
            throw std::runtime_error("HIP backend only supports HipCpp or HSACO source");
        }

        // Load HIP module from HSACO bytes
        // Diag patch (2026-05-12): also report sticky prior error + this call's rc + free VRAM
        hipError_t prior = hipGetLastError();   // also clears sticky state
        hipModule_t module;
        hipError_t loadRc = hipModuleLoadData(&module, hsacoBytes_.data());
        if (loadRc != hipSuccess) {
            size_t freeB = 0, totalB = 0;
            (void)hipMemGetInfo(&freeB, &totalB);
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "Failed to load HIP module from HSACO bytes [entry=%s rc=%d (%s) priorRc=%d (%s) hsacoSize=%zu freeVramMB=%zu]",
                source.entryPoint.c_str(),
                (int)loadRc, hipGetErrorString(loadRc),
                (int)prior, hipGetErrorString(prior),
                hsacoBytes_.size(),
                (size_t)(freeB / (1024*1024)));
            throw std::runtime_error(buf);
        }

        // Get kernel function
        if (hipModuleGetFunction(&kernel_, module, source.entryPoint.c_str()) != hipSuccess) {
            hipModuleUnload(module);
            throw std::runtime_error("Failed to get HIP kernel function");
        }

        module_ = module;
    }

    ~HipProgram() override {
        if (module_) {
            hipModuleUnload(module_);
        }
    }

    void* nativeHandle() override {
        return kernel_;
    }

    // Get compiled HSACO bytes for disk caching
    std::vector<std::uint8_t> getCompiledBytes() const override {
        return hsacoBytes_;
    }

    // Get compiled format identifier
    std::string getCompiledFormat() const override {
        return "hsaco";
    }

private:
    hipModule_t module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
    std::vector<std::uint8_t> hsacoBytes_;  // Store original HSACO bytes
};

// ============================================================================
// HipGraph - HIP graph capture and execution
// ============================================================================

class HipGraph : public Graph {
public:
    HipGraph() : graph_(nullptr), exec_(nullptr) {}

    ~HipGraph() override {
        if (exec_) {
            hipGraphExecDestroy(exec_);
        }
        if (graph_) {
            hipGraphDestroy(graph_);
        }
    }

    void launch(Queue& queue) override {
        if (!exec_) {
            throw std::runtime_error("Cannot launch uninstantiated HIP graph");
        }

        hipStream_t stream = static_cast<hipStream_t>(queue.nativeHandle());
        if (hipGraphLaunch(exec_, stream) != hipSuccess) {
            throw std::runtime_error("Failed to launch HIP graph");
        }
    }

    void update() override {
        // Update graph topology after parameter mutations
        // For now, just re-instantiate
        if (graph_ && exec_) {
            hipGraphExecDestroy(exec_);
            exec_ = nullptr;

            if (hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0) != hipSuccess) {
                throw std::runtime_error("Failed to re-instantiate HIP graph");
            }
        }
    }

    void beginCapture(Queue& queue) override {
        hipStream_t stream = static_cast<hipStream_t>(queue.nativeHandle());
        if (hipGraphCreate(&graph_, 0) != hipSuccess) {
            throw std::runtime_error("Failed to create HIP graph");
        }

        if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess) {
            hipGraphDestroy(graph_);
            graph_ = nullptr;
            throw std::runtime_error("Failed to begin HIP graph capture");
        }
    }

    void endCapture(Queue& queue) override {
        hipStream_t stream = static_cast<hipStream_t>(queue.nativeHandle());
        if (hipStreamEndCapture(stream, &graph_) != hipSuccess) {
            throw std::runtime_error("Failed to end HIP graph capture");
        }

        if (hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0) != hipSuccess) {
            throw std::runtime_error("Failed to instantiate HIP graph");
        }
    }

    hipGraph_t get() const { return graph_; }

private:
    hipGraph_t graph_;
    hipGraphExec_t exec_;
};

// ============================================================================
// HipDevice - HIP device implementation
// ============================================================================

class HipDevice : public Device {
public:
    HipDevice(int device_id) {
        if (hipSetDevice(device_id) != hipSuccess) {
            throw std::runtime_error("Failed to set HIP device");
        }

        if (hipGetDeviceProperties(&props_, device_id) != hipSuccess) {
            throw std::runtime_error("Failed to get HIP device properties");
        }

        device_id_ = device_id;
    }

    ~HipDevice() override = default;

    std::unique_ptr<Queue> createQueue(StreamPriority priority) override {
        return std::make_unique<HipQueue>(priority);
    }

    std::unique_ptr<Buffer> createBuffer(size_t size, MemoryKind kind) override {
        return std::make_unique<HipBuffer>(size, kind);
    }

    std::unique_ptr<Program> createProgram(const ProgramSource& source) override {
        return std::make_unique<HipProgram>(source, props_.gcnArchName);
    }

    // Graph creation
    std::unique_ptr<Graph> createGraph() override {
        return std::make_unique<HipGraph>();
    }

    // Event creation
    std::unique_ptr<Event> createEvent() override {
        return std::make_unique<HipEvent>();
    }

    // Device metadata
    std::string getName() const override {
        return props_.name;
    }

    size_t getTotalMemory() const override {
        return props_.totalGlobalMem;
    }

    std::string getComputeCapability() const override {
        // HIP uses gcnArchName instead of compute capability
        return props_.gcnArchName;
    }

    bool supportsFeature(const std::string& feature) const override {
        if (feature == "float64" || feature == "fp64") return true;
        if (feature == "float16" || feature == "fp16") return true;
        if (feature == "bfloat16" || feature == "bf16") return true;
        if (feature == "graphs") return true;  // HIP 5.0+
        if (feature == "managedMemory") return props_.managedMemory;
        if (feature == "sharedMem") return true;
        if (feature == "streams") return true;
        if (feature == "events") return true;
        if (feature == "tensor_cores") return true;  // Matrix cores (backend-specific mapping)
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
        // HIP doesn't have extensions like Vulkan
        if (extension == "hip") return true;
        if (extension == "hipComputeCapability") return true;
        return false;
    }

    uint32_t getSubgroupSize() const override {
        // Warp size for AMD GPUs
        return props_.warpSize;
    }

    bool supportsAtomicOp(AtomicOpType op, DataType type) const override {
        // HIP/AMD has good atomic support
        (void)op;

        switch (type) {
            case DataType::Float64:
                return true;  // AMD GPUs generally support atomic double
            case DataType::Float32:
                return true;
            case DataType::Float16:
            case DataType::BFloat16:
                return true;  // Modern AMD GPUs support atomic half
            case DataType::Int64:
            case DataType::UInt64:
            case DataType::Int32:
            case DataType::UInt32:
            case DataType::Int16:
            case DataType::UInt16:
            case DataType::Int8:
            case DataType::UInt8:
                return true;  // All integer atomics supported
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
        return {"hip"};
    }

    bool queryMemoryInfo(size_t* freeBytes, size_t* totalBytes) const override {
        size_t f = 0, t = 0;
        if (hipMemGetInfo(&f, &t) != hipSuccess) return false;
        if (freeBytes) *freeBytes = f;
        if (totalBytes) *totalBytes = t;
        return true;
    }

private:
    int device_id_;
    hipDeviceProp_t props_;
};

// Factory function
std::unique_ptr<Device> HipDevice_create(int device_id) {
    return std::make_unique<HipDevice>(device_id);
}

} // namespace GpuRuntime
