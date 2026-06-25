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

// GPU Runtime - OpenCL Backend Implementation
// Full OpenCL backend with Device, Queue, Buffer, Program, Event

#include <gpu_runtime.h>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <thread>
#include <cctype>
#include <optional>
#include <cstdlib>

namespace GpuRuntime {

namespace {

static std::string toLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static std::optional<int> envInt(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return std::nullopt;
    try {
        return std::stoi(v);
    } catch (...) {
        return std::nullopt;
    }
}

static std::optional<std::string> envStr(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return std::nullopt;
    return std::string(v);
}

static std::string clInfoString(cl_platform_id platform, cl_platform_info param) {
    size_t bytes = 0;
    if (clGetPlatformInfo(platform, param, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0) {
        return "";
    }
    std::string s(bytes, '\0');
    if (clGetPlatformInfo(platform, param, bytes, s.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static std::string clInfoString(cl_device_id device, cl_device_info param) {
    size_t bytes = 0;
    if (clGetDeviceInfo(device, param, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0) {
        return "";
    }
    std::string s(bytes, '\0');
    if (clGetDeviceInfo(device, param, bytes, s.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static cl_device_type parseDeviceType(std::optional<std::string> v) {
    if (!v) return CL_DEVICE_TYPE_ALL;
    std::string s = toLower(*v);
    if (s == "gpu") return CL_DEVICE_TYPE_GPU;
    if (s == "cpu") return CL_DEVICE_TYPE_CPU;
    if (s == "accel" || s == "accelerator") return CL_DEVICE_TYPE_ACCELERATOR;
    if (s == "default") return CL_DEVICE_TYPE_DEFAULT;
    if (s == "all") return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_ALL;
}

static int parseOpenclVersionMajor(const std::string& version) {
    // Typical strings: "OpenCL 1.2 ..." or "OpenCL 3.0 ..."
    auto pos = version.find("OpenCL ");
    if (pos == std::string::npos) return 1;
    pos += 7;
    if (pos >= version.size()) return 1;
    int major = 1;
    try {
        major = std::stoi(version.substr(pos));
    } catch (...) {
        major = 1;
    }
    return major;
}

static int parseOpenclVersionMinor(const std::string& version) {
    auto pos = version.find("OpenCL ");
    if (pos == std::string::npos) return 2;
    pos += 7;
    auto dot = version.find('.', pos);
    if (dot == std::string::npos) return 0;
    try {
        return std::stoi(version.substr(dot + 1));
    } catch (...) {
        return 0;
    }
}

struct OpenclCandidate {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string platformName;
    std::string deviceName;
    cl_device_type deviceType = 0;
    cl_uint computeUnits = 0;
};

static std::vector<OpenclCandidate> enumerateCandidates(
    cl_device_type deviceTypeFilter,
    const std::optional<int>& platformIndexFilter,
    const std::optional<std::string>& platformNameFilter,
    const std::optional<std::string>& deviceNameFilter
) {
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return {};
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return {};
    }

    std::vector<OpenclCandidate> candidates;
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_platform_id platform = platforms[pi];
        std::string pname = clInfoString(platform, CL_PLATFORM_NAME);
        std::string pnameLower = toLower(pname);

        if (platformIndexFilter && static_cast<int>(pi) != *platformIndexFilter) {
            continue;
        }
        if (platformNameFilter) {
            std::string needle = toLower(*platformNameFilter);
            if (!needle.empty() && pnameLower.find(needle) == std::string::npos) {
                continue;
            }
        }

        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platform, deviceTypeFilter, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(num_devices);
        if (clGetDeviceIDs(platform, deviceTypeFilter, num_devices, devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        for (cl_uint di = 0; di < num_devices; ++di) {
            cl_device_id dev = devices[di];
            OpenclCandidate c;
            c.platform = platform;
            c.device = dev;
            c.platformName = pname;
            c.deviceName = clInfoString(dev, CL_DEVICE_NAME);
            clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(c.deviceType), &c.deviceType, nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(c.computeUnits), &c.computeUnits, nullptr);

            if (deviceNameFilter) {
                std::string needle = toLower(*deviceNameFilter);
                if (!needle.empty() && toLower(c.deviceName).find(needle) == std::string::npos) {
                    continue;
                }
            }

            candidates.push_back(std::move(c));
        }
    }

    return candidates;
}

static std::string describeCandidates(const std::vector<OpenclCandidate>& candidates) {
    std::string out;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        out += "  [" + std::to_string(i) + "] platform=\"" + c.platformName + "\" device=\"" + c.deviceName + "\"";
        if (c.deviceType & CL_DEVICE_TYPE_GPU) out += " type=gpu";
        else if (c.deviceType & CL_DEVICE_TYPE_CPU) out += " type=cpu";
        else if (c.deviceType & CL_DEVICE_TYPE_ACCELERATOR) out += " type=accel";
        else out += " type=other";
        out += " computeUnits=" + std::to_string(c.computeUnits) + "\n";
    }
    return out;
}

}  // namespace

// Forward declarations for helpers that need OpenCLBuffer (defined later).
static void openclEnsureUnmappedForDevice(Buffer& b);
static void openclEnsureUnmappedForDevice(const Buffer& b);

// ============================================================================
// OpenCLEvent - Native OpenCL event implementation
// ============================================================================

class OpenCLEvent : public Event {
public:
    OpenCLEvent(cl_event event) : event_(event) {}

    ~OpenCLEvent() override {
        if (event_) {
            clReleaseEvent(event_);
        }
    }

    void reset(cl_event newEvent) {
        if (event_) {
            clReleaseEvent(event_);
        }
        event_ = newEvent;
    }

    bool isComplete() const override {
        if (!event_) return true;
        cl_int status;
        cl_int err = clGetEventInfo(event_, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                     sizeof(status), &status, nullptr);
        if (err != CL_SUCCESS) return false;
        return status == CL_COMPLETE;
    }

    void wait() override {
        if (!event_) return;
        if (clWaitForEvents(1, &event_) != CL_SUCCESS) {
            throw std::runtime_error("Failed to wait for OpenCL event");
        }
    }

    void* nativeHandle() override {
        return reinterpret_cast<void*>(event_);
    }

    cl_event get() const { return event_; }

private:
    cl_event event_;
};

// ============================================================================
// OpenCLQueue - OpenCL command queue implementation
// ============================================================================

class OpenCLQueue : public Queue {
public:
    OpenCLQueue(cl_command_queue queue) : queue_(queue) {}

    ~OpenCLQueue() override {
        if (queue_) {
            clReleaseCommandQueue(queue_);
        }
    }

    void submit(Program& program, const std::vector<Buffer*>& args,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z) override {
        // Get OpenCL kernel from program
        cl_kernel kernel = static_cast<cl_kernel>(program.nativeHandle());

        const bool validateArgs = []() -> bool {
            const char* p = std::getenv("GPU_RUNTIME_OPENCL_VALIDATE_ARGS");
            if (!p || p[0] == '\0') return false;
            return std::string(p) != "0";
        }();

        if (validateArgs)
        {
            cl_uint kernelNumArgs = 0;
            (void)clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernelNumArgs), &kernelNumArgs, nullptr);
            if (kernelNumArgs != 0 && args.size() != static_cast<size_t>(kernelNumArgs))
            {
                size_t n = 0;
                (void)clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &n);
                std::string name;
                if (n > 0)
                {
                    name.resize(n);
                    if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, n, name.data(), nullptr) == CL_SUCCESS)
                    {
                        if (!name.empty() && name.back() == '\0') name.pop_back();
                    }
                    else
                    {
                        name = "<unknown-kernel>";
                    }
                }
                else
                {
                    name = "<unknown-kernel>";
                }

                throw std::runtime_error(
                    "OpenCL kernel arg count mismatch: kernel=" + name +
                    " expected=" + std::to_string(kernelNumArgs) +
                    " got=" + std::to_string(args.size()));
            }
        }

        // Set kernel arguments
        for (size_t i = 0; i < args.size(); ++i) {
            // If the buffer is host-mapped, unmap it before device access.
            openclEnsureUnmappedForDevice(*args[i]);
            cl_mem buf = reinterpret_cast<cl_mem>(args[i]->nativeHandle());
            const cl_int rc = clSetKernelArg(kernel, i, sizeof(cl_mem), &buf);
            if (rc != CL_SUCCESS) {
                size_t n = 0;
                (void)clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &n);
                std::string name;
                if (n > 0)
                {
                    name.resize(n);
                    if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, n, name.data(), nullptr) == CL_SUCCESS)
                    {
                        if (!name.empty() && name.back() == '\0') name.pop_back();
                    }
                    else
                    {
                        name = "<unknown-kernel>";
                    }
                }
                else
                {
                    name = "<unknown-kernel>";
                }

                cl_uint kernelNumArgs = 0;
                (void)clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernelNumArgs), &kernelNumArgs, nullptr);

                throw std::runtime_error(
                    "Failed to set OpenCL kernel argument: kernel=" + name +
                    " expectedArgs=" + std::to_string(kernelNumArgs) +
                    " gotArgs=" + std::to_string(args.size()) +
                    " argIndex=" + std::to_string(i) +
                    " rc=" + std::to_string(rc));
            }
        }

        // Set work dimensions
        size_t global_work_size[3] = {grid_x * block_x, grid_y * block_y, grid_z * block_z};
        size_t local_work_size[3] = {block_x, block_y, block_z};

        // Determine work dimensions based on actual usage
        cl_uint work_dim = 1;
        if (grid_y > 1 || block_y > 1) {
            work_dim = 2;
        }
        if (grid_z > 1 || block_z > 1) {
            work_dim = 3;
        }

        if (clEnqueueNDRangeKernel(queue_, kernel, work_dim, nullptr,
                                    global_work_size, local_work_size,
                                    0, nullptr, nullptr) != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue OpenCL kernel");
        }
    }

    void memcpy(Buffer& dst, const Buffer& src, size_t size) override {
        memcpyOffset(dst, 0, src, 0, size);
    }

    void memcpyOffset(Buffer& dst, size_t dstOffset,
                      const Buffer& src, size_t srcOffset,
                      size_t size) override {
        if (dstOffset > dst.sizeBytes() || size > dst.sizeBytes() - dstOffset ||
            srcOffset > src.sizeBytes() || size > src.sizeBytes() - srcOffset) {
            throw std::runtime_error("OpenCL memcpyOffset out of bounds");
        }
        // Ensure host-mapped buffers are unmapped before device-side operations.
        openclEnsureUnmappedForDevice(dst);
        openclEnsureUnmappedForDevice(src);
        cl_mem dstBuf = reinterpret_cast<cl_mem>(dst.nativeHandle());
        cl_mem srcBuf = reinterpret_cast<cl_mem>(src.nativeHandle());

        if (clEnqueueCopyBuffer(queue_, srcBuf, dstBuf, srcOffset, dstOffset, size,
                                  0, nullptr, nullptr) != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue OpenCL memcpy");
        }
    }

    void synchronize() override {
        if (clFinish(queue_) != CL_SUCCESS) {
            throw std::runtime_error("Failed to synchronize OpenCL queue");
        }
    }

    void* nativeHandle() override {
        return queue_;
    }

    // Event recording
    void record(Event& event) override {
        OpenCLEvent* clEvent = static_cast<OpenCLEvent*>(&event);
        cl_event ev = nullptr;
        cl_int err = clEnqueueMarkerWithWaitList(queue_, 0, nullptr, &ev);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to record OpenCL event");
        }
        // Ensure the marker is submitted so it can make progress for polling waits.
        clFlush(queue_);
        clEvent->reset(ev);  // takes ownership of the event reference
    }

    // Wait for event
    void wait(Event& event) override {
        OpenCLEvent* clEvent = static_cast<OpenCLEvent*>(&event);
        cl_event ev = clEvent->get();
        if (!ev) return;
        // Insert a barrier so subsequent commands in this queue wait for `ev`.
        cl_int err = clEnqueueBarrierWithWaitList(queue_, 1, &ev, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue OpenCL event wait");
        }
    }

    // Wait with timeout
    bool waitFor(Event& event, uint64_t timeout_us) override {
        OpenCLEvent* clEvent = static_cast<OpenCLEvent*>(&event);
        cl_event ev = clEvent->get();
        if (!ev) return true;
        // Ensure queued work is submitted so the event can make progress.
        clFlush(queue_);
        cl_int status = CL_QUEUED;
        cl_int err = clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(status), &status, nullptr);
        if (err == CL_SUCCESS && status == CL_COMPLETE) return true;

        // OpenCL doesn't have a timeout wait for events
        // We'll use polling with sleep
        auto start = std::chrono::steady_clock::now();
        while (true) {
            cl_int curStatus = CL_QUEUED;
            cl_int curErr = clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                           sizeof(curStatus), &curStatus, nullptr);
            if (curErr == CL_SUCCESS && curStatus == CL_COMPLETE) return true;

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
            if (elapsed.count() >= timeout_us) return false;

            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Multi-queue synchronization
    void wait(Event& event, Queue& other) override {
        // OpenCL events work across queues by default
        (void)other;
        wait(event);
    }

private:
    cl_command_queue queue_;
};

// ============================================================================
// OpenCLBuffer - OpenCL device memory buffer
// ============================================================================

class OpenCLBuffer : public Buffer {
public:
    OpenCLBuffer(cl_context context, cl_command_queue queue, size_t size, MemoryKind kind)
        : context_(context), queue_(queue), size_(size), kind_(kind) {

        cl_mem_flags flags;
        switch (kind) {
            case MemoryKind::Device:
                flags = CL_MEM_READ_WRITE;
                hostPtr_ = nullptr;
                break;
            case MemoryKind::HostPinned:
                flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE;
                break;
            case MemoryKind::Managed:
                flags = CL_MEM_READ_WRITE;  // OpenCL doesn't have managed memory like CUDA
                hostPtr_ = nullptr;
                break;
        }

        buffer_ = clCreateBuffer(context_, flags, size, nullptr, nullptr);
        if (!buffer_) {
            throw std::runtime_error("Failed to create OpenCL buffer");
        }
        hostPtr_ = nullptr;
    }

    ~OpenCLBuffer() override {
        ensureUnmapped();
        if (buffer_) {
            clReleaseMemObject(buffer_);
        }
    }

    size_t sizeBytes() const override { return size_; }
    MemoryKind memoryKind() const override { return kind_; }

    void* getDevicePointer() override {
        return reinterpret_cast<void*>(buffer_);
    }

    void* getHostPointer() override {
        // For OpenCL, host visibility depends on mapping.
        // This runtime treats `HostPinned` buffers as "implicitly mappable": the first call maps it,
        // and device operations will auto-unmap it for coherence.
        if (kind_ == MemoryKind::Device) {
            return nullptr;
        }
        if (!hostPtr_) {
            // Default to a read/write mapping; this mirrors "host-visible" semantics used in tests.
            hostPtr_ = clEnqueueMapBuffer(queue_, buffer_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                          0, size_, 0, nullptr, nullptr, nullptr);
            if (!hostPtr_) {
                throw std::runtime_error("Failed to map OpenCL buffer (implicit getHostPointer)");
            }
        }
        return hostPtr_;
    }

    void map(MapMode mode) override {
        if (kind_ == MemoryKind::Device) {
            throw std::runtime_error("Cannot map OpenCL device-only buffer");
        }

        // If already mapped, unmap first to avoid driver-side incoherence across map modes.
        ensureUnmapped();

        cl_map_flags flags;
        switch (mode) {
            case MapMode::Read:
                flags = CL_MAP_READ;
                break;
            case MapMode::Write:
                flags = CL_MAP_WRITE;
                break;
            case MapMode::ReadWrite:
                flags = CL_MAP_READ | CL_MAP_WRITE;
                break;
        }

        hostPtr_ = clEnqueueMapBuffer(queue_, buffer_, CL_TRUE, flags,
                                      0, size_, 0, nullptr, nullptr, nullptr);
        if (!hostPtr_) {
            throw std::runtime_error("Failed to map OpenCL buffer");
        }
    }

    void unmap() override {
        ensureUnmapped();
    }

    void ensureUnmapped() {
        if (!hostPtr_ || kind_ == MemoryKind::Device) {
            return;
        }
        if (!queue_) {
            // Should not happen for host-visible buffer kinds; keep defensive.
            hostPtr_ = nullptr;
            return;
        }
        cl_int err = clEnqueueUnmapMemObject(queue_, buffer_, hostPtr_, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to unmap OpenCL buffer");
        }
        // Ensure the unmap completes before any subsequent device command uses this buffer.
        clFinish(queue_);
        hostPtr_ = nullptr;
    }

    void* nativeHandle() const override {
        return reinterpret_cast<void*>(buffer_);
    }

private:
    cl_context context_;
    cl_command_queue queue_;
    size_t size_;
    MemoryKind kind_;
    cl_mem buffer_;
    void* hostPtr_;
};

static void openclEnsureUnmappedForDevice(Buffer& b) {
    if (auto* ob = dynamic_cast<OpenCLBuffer*>(&b)) {
        ob->ensureUnmapped();
    }
}

static void openclEnsureUnmappedForDevice(const Buffer& b) {
    openclEnsureUnmappedForDevice(const_cast<Buffer&>(b));
}

// ============================================================================
// OpenCLProgram - Compiled OpenCL kernel program
// ============================================================================

class OpenCLProgram : public Program {
public:
    OpenCLProgram(cl_context context, cl_device_id device, const ProgramSource& source)
        : context_(context), device_(device), program_(nullptr), kernel_(nullptr) {

        if (source.kind != ProgramSource::Kind::OpenClC &&
            source.kind != ProgramSource::Kind::OpenClBinary &&
            source.kind != ProgramSource::Kind::Spirv) {
            throw std::runtime_error("OpenCL backend only supports OpenCL C, OpenCL program binaries, or SPIR-V IL");
        }

        compiledFormat_ = "unknown";

        cl_uint num_devices = 1;
        cl_device_id devices[] = {device};

        if (source.kind == ProgramSource::Kind::Spirv) {
            // Load from SPIR-V IL (OpenCL 2.1+). If the ICD does not support this, it must fail.
            cl_int err = CL_SUCCESS;
            program_ = clCreateProgramWithIL(context_, source.bytes.data(), source.bytes.size(), &err);
            if (err != CL_SUCCESS || !program_) {
                throw std::runtime_error("Failed to create OpenCL program from SPIR-V IL (requires OpenCL 2.1+)");
            }
            compiledFormat_ = "spv";
        } else if (source.kind == ProgramSource::Kind::OpenClBinary) {
            // Load from an OpenCL program binary produced by a previous build on the same device/driver.
            const unsigned char* binary = reinterpret_cast<const unsigned char*>(source.bytes.data());
            size_t binarySize = source.bytes.size();
            cl_int binary_status = CL_SUCCESS;
            program_ = clCreateProgramWithBinary(context_, num_devices, devices,
                                                 &binarySize, &binary, &binary_status, nullptr);
            if (binary_status != CL_SUCCESS || !program_) {
                throw std::runtime_error("Failed to create OpenCL program from cached binary");
            }
            compiledBytes_ = source.bytes;
            compiledFormat_ = "opencl_bin";
        } else {
            // Compile from OpenCL C source
            const char* source_str = reinterpret_cast<const char*>(source.bytes.data());
            size_t source_size = source.bytes.size();
            program_ = clCreateProgramWithSource(context_, 1, &source_str, &source_size, nullptr);
            if (!program_) {
                throw std::runtime_error("Failed to create OpenCL program from source");
            }

            // OpenCL compiler option policy:
            // - Default to CL1.2 for maximum ICD compatibility/stability.
            // - Allow overriding with GPU_RUNTIME_OPENCL_CL_STD=CL1.2|CL2.0|CL3.0|none.
            std::string buildOpts = "-cl-std=CL1.2";
            if (auto stdOverride = envStr("GPU_RUNTIME_OPENCL_CL_STD")) {
                const std::string v = toLower(*stdOverride);
                if (v == "none" || v == "0" || v == "false") {
                    buildOpts.clear();
                } else if (v == "cl1.2" || v == "1.2") {
                    buildOpts = "-cl-std=CL1.2";
                } else if (v == "cl2.0" || v == "2.0") {
                    buildOpts = "-cl-std=CL2.0";
                } else if (v == "cl3.0" || v == "3.0") {
                    buildOpts = "-cl-std=CL3.0";
                }
            }

            auto buildWithOpts = [&](const char* opts) -> cl_int {
                return clBuildProgram(program_, 1, &device, opts, nullptr, nullptr);
            };

            cl_int buildErr = buildWithOpts(buildOpts.c_str());
            if (buildErr != CL_SUCCESS) {
                // Some ICDs reject `-cl-std=` even though they support the language; retry without options.
                buildErr = buildWithOpts("");
            }
            if (buildErr != CL_SUCCESS) {
                // Get build log
                size_t log_size = 0;
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::string log;
                if (log_size > 1) {
                    log.resize(log_size, '\0');
                    clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                } else {
                    log = "(no build log)";
                }
                clReleaseProgram(program_);
                throw std::runtime_error("Failed to build OpenCL program (" + buildOpts + "): " + log);
            }

            // Extract per-device program binary (for persistent disk cache).
            size_t binarySize = 0;
            cl_int sizeErr = clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, nullptr);
            if (sizeErr == CL_SUCCESS && binarySize > 0) {
                std::vector<std::uint8_t> bin(binarySize);
                unsigned char* binPtr = reinterpret_cast<unsigned char*>(bin.data());
                unsigned char* binaries[] = {binPtr};
                cl_int binErr = clGetProgramInfo(program_, CL_PROGRAM_BINARIES, sizeof(binaries), binaries, nullptr);
                if (binErr == CL_SUCCESS) {
                    compiledBytes_ = std::move(bin);
                    compiledFormat_ = "opencl_bin";
                } else {
                    compiledFormat_ = "cl";  // cannot extract compiled bytes; don't disk-cache
                }
            } else {
                compiledFormat_ = "cl";  // cannot extract compiled bytes; don't disk-cache
            }
        }

        // Build program (OpenClBinary/Spirv still needs build/validation step).
        if (source.kind != ProgramSource::Kind::OpenClC) {
            if (clBuildProgram(program_, num_devices, devices, nullptr, nullptr, nullptr) != CL_SUCCESS) {
                size_t log_size = 0;
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::string log;
                if (log_size > 1) {
                    log.resize(log_size, '\0');
                    clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                } else {
                    log = "(no build log)";
                }
                clReleaseProgram(program_);
                throw std::runtime_error("Failed to build OpenCL program: " + log);
            }
        }

        // Create kernel
        kernel_ = clCreateKernel(program_, source.entryPoint.c_str(), nullptr);
        if (!kernel_) {
            clReleaseProgram(program_);
            throw std::runtime_error("Failed to create OpenCL kernel: " + source.entryPoint);
        }
    }

    ~OpenCLProgram() override {
        if (kernel_) {
            clReleaseKernel(kernel_);
        }
        if (program_) {
            clReleaseProgram(program_);
        }
    }

    void* nativeHandle() override {
        return kernel_;
    }

    // Get compiled bytes for disk caching
    std::vector<std::uint8_t> getCompiledBytes() const override {
        if (!compiledBytes_.empty()) {
            return compiledBytes_;
        }
        return {};  // no compiled artifact available
    }

    // Get compiled format identifier
    std::string getCompiledFormat() const override {
        return compiledFormat_;
    }

private:
    cl_context context_;
    cl_device_id device_;
    cl_program program_;
    cl_kernel kernel_;
    std::vector<std::uint8_t> compiledBytes_;
    std::string compiledFormat_;
};

// ============================================================================
// OpenCLDevice - OpenCL device implementation
// ============================================================================

class OpenCLDevice : public Device {
public:
    OpenCLDevice(int device_id) {
        const auto platformIndexFilter = envInt("GPU_RUNTIME_OPENCL_PLATFORM_INDEX");
        const auto platformNameFilter = envStr("GPU_RUNTIME_OPENCL_PLATFORM_NAME");
        const auto deviceNameFilter = envStr("GPU_RUNTIME_OPENCL_DEVICE_NAME");
        const auto deviceIndexOverride = envInt("GPU_RUNTIME_OPENCL_DEVICE_INDEX");
        const auto deviceTypeEnv = envStr("GPU_RUNTIME_OPENCL_DEVICE_TYPE");
        const auto deviceTypeFilter = parseDeviceType(deviceTypeEnv);

        // Enumerate candidates (filtered by env vars).
        auto candidates = enumerateCandidates(deviceTypeFilter, platformIndexFilter, platformNameFilter, deviceNameFilter);
        if (candidates.empty() && !deviceTypeEnv && deviceTypeFilter != CL_DEVICE_TYPE_ALL) {
            // Best-effort fallback: widen to ALL if the filter is too strict.
            candidates = enumerateCandidates(CL_DEVICE_TYPE_ALL, platformIndexFilter, platformNameFilter, deviceNameFilter);
        }
        if (candidates.empty()) {
            throw std::runtime_error(
                "No OpenCL devices found after applying filters. "
                "Set GPU_RUNTIME_OPENCL_PLATFORM_INDEX / GPU_RUNTIME_OPENCL_PLATFORM_NAME and "
                "GPU_RUNTIME_OPENCL_DEVICE_INDEX / GPU_RUNTIME_OPENCL_DEVICE_NAME / GPU_RUNTIME_OPENCL_DEVICE_TYPE."
            );
        }

        int chosenIndex = 0;
        if (deviceIndexOverride) {
            chosenIndex = *deviceIndexOverride;
        } else if (device_id >= 0) {
            chosenIndex = device_id;
        }
        if (chosenIndex < 0 || static_cast<size_t>(chosenIndex) >= candidates.size()) {
            throw std::runtime_error(
                "Invalid OpenCL device selection index. Available devices:\n" + describeCandidates(candidates)
            );
        }

        platform_ = candidates[chosenIndex].platform;
        device_ = candidates[chosenIndex].device;

        // Create context tied to the selected platform (more robust than nullptr properties).
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform_),
            0
        };
        context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, nullptr);
        if (!context_) {
            throw std::runtime_error("Failed to create OpenCL context");
        }

        // Dedicated queue for buffer mapping/unmapping.
        // Avoid creating a command queue per buffer: some ICDs/drivers (notably ROCm OpenCL)
        // can become unstable/resource-exhausted when many queues are created during startup.
        cl_int qErr = CL_SUCCESS;
        mapQueue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &qErr);
        if (qErr != CL_SUCCESS || !mapQueue_) {
            clReleaseContext(context_);
            context_ = nullptr;
            throw std::runtime_error("Failed to create OpenCL command queue for buffer mapping");
        }

        // Get device properties
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name_), name_, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &totalMem_, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits_, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize_, nullptr);

        // Check for extensions
        size_t ext_size;
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
        extensions_.resize(ext_size);
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_size, extensions_.data(), nullptr);
    }

    ~OpenCLDevice() override {
        if (mapQueue_) {
            clReleaseCommandQueue(mapQueue_);
        }
        if (context_) {
            clReleaseContext(context_);
        }
    }

    std::unique_ptr<Queue> createQueue(StreamPriority priority) override {
        // OpenCL doesn't have stream priorities - ignore priority parameter
        (void)priority;

        // Use a single shared in-order queue for both mapping and compute.
        //
        // Rationale:
        // - Buffers in this runtime use a dedicated queue for map/unmap.
        // - When compute uses a different queue, some OpenCL stacks (notably ROCm)
        //   can exhibit coherence issues without explicit cross-queue event wiring.
        // - Sharing a single queue keeps ordering/coherence simple and deterministic.
        //
        // We retain the command queue so each OpenCLQueue instance owns a reference.
        if (!mapQueue_) {
            throw std::runtime_error("OpenCL mapping queue not initialized");
        }
        clRetainCommandQueue(mapQueue_);
        return std::make_unique<OpenCLQueue>(mapQueue_);
    }

    std::unique_ptr<Buffer> createBuffer(size_t size, MemoryKind kind) override {
        return std::make_unique<OpenCLBuffer>(context_, mapQueue_, size, kind);
    }

    std::unique_ptr<Program> createProgram(const ProgramSource& source) override {
        return std::make_unique<OpenCLProgram>(context_, device_, source);
    }

    // Graph creation (not supported for OpenCL)
    std::unique_ptr<Graph> createGraph() override {
        throw std::runtime_error("OpenCL backend does not support graphs");
    }

    // Event creation
    std::unique_ptr<Event> createEvent() override {
        // OpenCL events are created when enqueueing commands (e.g. markers/barriers/kernels).
        // Returning an "empty" event avoids driver quirks around user events that are never signaled.
        return std::make_unique<OpenCLEvent>(nullptr);
    }

    // Device metadata
    std::string getName() const override {
        return name_;
    }

    size_t getTotalMemory() const override {
        return totalMem_;
    }

    std::string getComputeCapability() const override {
        // OpenCL doesn't have compute capability like CUDA
        // Return device version
        size_t version_size;
        clGetDeviceInfo(device_, CL_DEVICE_VERSION, 0, nullptr, &version_size);
        std::string version(version_size, '\0');
        clGetDeviceInfo(device_, CL_DEVICE_VERSION, version_size, &version[0], nullptr);
        return version;
    }

    bool supportsFeature(const std::string& feature) const override {
        if (feature == "float64" || feature == "fp64") {
            // Check cl_khr_fp64 extension
            return extensions_.find("cl_khr_fp64") != std::string::npos;
        }
        if (feature == "float16" || feature == "fp16") {
            // Check cl_khr_fp16 extension
            return extensions_.find("cl_khr_fp16") != std::string::npos;
        }
        if (feature == "bfloat16" || feature == "bf16") return false;
        if (feature == "graphs") return false;  // OpenCL doesn't have graphs
        if (feature == "managedMemory") return false;  // OpenCL doesn't have managed memory
        if (feature == "sharedMem") return true;
        if (feature == "streams") return true;
        if (feature == "events") return true;
        if (feature == "tensor_cores") return false;
        if (feature == "memory_pools") return true;
        if (feature == "nurbs.rational.evaluate") {
            return extensions_.find("cl_khr_fp64") != std::string::npos;
        }
        if (feature == "nurbs.rational.derivatives") {
            return extensions_.find("cl_khr_fp64") != std::string::npos;
        }
        if (feature == "nurbs.rational.project") {
            return extensions_.find("cl_khr_fp64") != std::string::npos;
        }
        return false;
    }

    // Extended capability queries
    DeviceVersion getVersion() const override {
        // Parse OpenCL version string
        std::string version = getComputeCapability();
        int major = parseOpenclVersionMajor(version);
        int minor = parseOpenclVersionMinor(version);
        return DeviceVersion{major, minor};
    }

    bool hasExtension(const std::string& extension) const override {
        return extensions_.find(extension) != std::string::npos;
    }

	    uint32_t getSubgroupSize() const override {
	        // NOTE: OpenCL core does not expose a single fixed “subgroup size” on cl_device_id.
	        //
	        // For gpu_runtime we still need a non-zero "warp/subgroup" concept for:
	        // - selecting reasonable defaults for reductions/scans
	        // - avoiding divide-by-zero/undefined behavior in backends that use subgroup size
	        //
	        // Strategy:
	        //  1) Allow explicit override via env var (most robust).
	        //  2) Query vendor-specific extensions where available.
	        //  3) Fall back to a conservative heuristic based on vendor string.
	        //
	        // Override:
	        //   GPU_RUNTIME_OPENCL_SUBGROUP_SIZE=<int>
	        if (auto override = envInt("GPU_RUNTIME_OPENCL_SUBGROUP_SIZE")) {
	            if (*override > 0) return static_cast<uint32_t>(*override);
	        }

	        // Vendor-specific queries.
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
	        if (extensions_.find("cl_amd_device_attribute_query") != std::string::npos) {
	            cl_uint wavefront = 0;
	            if (clGetDeviceInfo(device_, CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(wavefront), &wavefront, nullptr) == CL_SUCCESS) {
	                if (wavefront > 0) return static_cast<uint32_t>(wavefront);
	            }
	        }
#endif
#ifdef CL_DEVICE_WARP_SIZE_NV
	        if (extensions_.find("cl_nv_device_attribute_query") != std::string::npos) {
	            cl_uint warp = 0;
	            if (clGetDeviceInfo(device_, CL_DEVICE_WARP_SIZE_NV, sizeof(warp), &warp, nullptr) == CL_SUCCESS) {
	                if (warp > 0) return static_cast<uint32_t>(warp);
	            }
	        }
#endif

	        // Some vendors expose it via Intel-specific extensions, e.g. cl_intel_subgroups.
#ifdef CL_DEVICE_SUB_GROUP_SIZES_INTEL
	        if (extensions_.find("cl_intel_subgroups") != std::string::npos) {
	            size_t bytes = 0;
	            if (clGetDeviceInfo(device_, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, nullptr, &bytes) != CL_SUCCESS) {
	                return 0;
            }
            if (bytes < sizeof(size_t)) return 0;
            std::vector<size_t> sizes(bytes / sizeof(size_t));
            if (clGetDeviceInfo(device_, CL_DEVICE_SUB_GROUP_SIZES_INTEL, bytes, sizes.data(), nullptr) != CL_SUCCESS) {
                return 0;
            }
	            if (!sizes.empty()) return static_cast<uint32_t>(sizes[0]);
	        }
#endif

	        // Heuristic fallback (correctness-first).
	        // We prefer returning a typical power-of-two subgroup size rather than 0.
	        const std::string vendor = toLower(clInfoString(device_, CL_DEVICE_VENDOR));
	        if (vendor.find("advanced micro devices") != std::string::npos || vendor.find("amd") != std::string::npos) {
	            return 64;
	        }
	        if (vendor.find("nvidia") != std::string::npos) {
	            return 32;
	        }
	        if (vendor.find("intel") != std::string::npos) {
	            return 16;
	        }
	        return 32;
	    }

    bool supportsAtomicOp(AtomicOpType op, DataType type) const override {
        (void)op;

        // NOTE: OpenCL atomics are subtle; be conservative in capability reports.
        // - Integer atomics are widely supported.
        // - Float atomics are typically extension-gated (e.g. cl_ext_float_atomics).
        const bool hasFloatAtomics =
            extensions_.find("cl_ext_float_atomics") != std::string::npos ||
            extensions_.find("cl_khr_fp32_atomic") != std::string::npos;

        switch (type) {
            case DataType::Float64:
                return extensions_.find("cl_khr_fp64") != std::string::npos;
            case DataType::Float32:
                return hasFloatAtomics;
            case DataType::Float16:
            case DataType::BFloat16:
                return false;  // OpenCL doesn't support atomic half
            case DataType::Int64:
            case DataType::UInt64:
                return extensions_.find("cl_khr_int64_base_atomics") != std::string::npos;
            case DataType::Int32:
            case DataType::UInt32:
                return true;  // OpenCL 1.0+ supports 32-bit atomics
            case DataType::Int16:
            case DataType::UInt16:
            case DataType::Int8:
            case DataType::UInt8:
                return false;  // OpenCL doesn't support small integer atomics
        }
        return false;
    }

    size_t getMaxSharedMemoryPerWorkgroup() const override {
        size_t local_mem_size;
        if (clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_t),
                             &local_mem_size, nullptr) == CL_SUCCESS) {
            return local_mem_size;
        }
        return 32 * 1024;  // Default fallback
    }

    uint32_t getMaxWorkgroupSize() const override {
        return static_cast<uint32_t>(maxWorkGroupSize_);
    }

    std::vector<std::string> getSupportedExtensions() const override {
        // Parse extensions string into individual extensions
        std::vector<std::string> result;
        std::string ext = extensions_;
        size_t start = 0;
        while (true) {
            size_t end = ext.find(' ', start);
            if (end == std::string::npos) {
                result.push_back(ext.substr(start));
                break;
            }
            result.push_back(ext.substr(start, end - start));
            start = end + 1;
        }
        return result;
    }

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue mapQueue_ = nullptr;
    char name_[256];
    size_t totalMem_;
    cl_uint computeUnits_;
    size_t maxWorkGroupSize_;
    std::string extensions_;
};

// Factory function (returns nullptr if OpenCL not available)
std::unique_ptr<Device> OpenCLDevice_tryCreate(int device_id) {
    try {
        // Try to create OpenCL device
        return std::make_unique<OpenCLDevice>(device_id);
    } catch (const std::exception& e) {
        std::cerr << "OpenCL backend unavailable: " << e.what() << "\n";
        return nullptr;
    }
}

} // namespace GpuRuntime
