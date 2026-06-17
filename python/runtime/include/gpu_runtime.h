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

// GPU Runtime - Cross-platform GPU compute runtime interface
// This header defines the stable public API for the gpu_runtime library
// Backend implementations: CUDA, HIP, Metal, Vulkan, OpenCL

#pragma once
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace GpuRuntime {

class Queue;
class Buffer;
class Event;
class Graph;

// ============================================================================
// Program Source (compiled kernel binary)
// ============================================================================

struct ProgramSource {
    enum class Kind {
        CudaCpp, HipCpp, OpenClC, OpenClBinary, Glsl, Spirv, Msl,
        Ptx, Cubin, Hsaco, Metallib
    };
    Kind kind;
    std::vector<std::uint8_t> bytes;
    std::string entryPoint;
    // Optional precompiled pipeline cache (Metal only; ignored by other backends).
    // When non-empty alongside an Msl source, the Metal backend loads the compute
    // pipeline state from this serialized MTLBinaryArchive instead of JIT-compiling
    // it — bypassing the macOS Metal shader-compiler FSCache flock() contention that
    // wedges multi-rank startup (Bug 7). Empty => compile from source and expose the
    // freshly serialized archive via Program::getCompiledBytes() (format
    // "metalarchive") so the disk cache can persist it for subsequent loads.
    std::vector<std::uint8_t> binaryArchive;
};

class Program {
public:
    virtual ~Program() = default;

    // Get backend handle (CUprogram, hipModule_t, cl_program, etc.)
    virtual void* nativeHandle() = 0;

    // Get compiled bytecode for disk caching
    // Returns the original compiled bytes (PTX, HSACO, SPIR-V, metallib, etc.)
    virtual std::vector<std::uint8_t> getCompiledBytes() const = 0;

    // Get compiled format identifier ("ptx", "hsaco", "spv", "metallib", etc.)
    virtual std::string getCompiledFormat() const = 0;
};

// Device-buffer memory telemetry (NAVATALA_GPU_MEMORY_PROFILE). On Apple Silicon
// (unified memory) device buffers are MTLStorageModeShared and live in system
// RAM, so currentBytes is the GPU-side contribution to host memory pressure.
// Backends that do not track allocations return all-zero.
struct DeviceMemTelemetry {
    std::size_t currentBytes = 0;   // live device-buffer bytes right now
    std::size_t peakBytes = 0;      // high-water of currentBytes
    std::size_t count = 0;          // live device-buffer count
    std::size_t largestBytes = 0;   // largest single device buffer ever allocated
};
DeviceMemTelemetry deviceMemTelemetry();

// ============================================================================
// Event (Native event objects for multi-queue sync)
// ============================================================================

class Event {
public:
    virtual ~Event() = default;

    // Non-blocking completion query
    virtual bool isComplete() const = 0;

    // Blocking wait
    virtual void wait() = 0;

    // Get backend handle (cudaEvent_t*, cl_event, VkEvent, etc.)
    virtual void* nativeHandle() = 0;
};

// ============================================================================
// Graph (CUDA/HIP graph capture and execution)
// ============================================================================

class Graph {
public:
    virtual ~Graph() = default;

    // Capture operations submitted to a queue into this graph (CUDA/HIP only).
    // Typical usage:
    //   graph->beginCapture(queue);
    //   ... submit kernels/memcpys/events on the same queue ...
    //   graph->endCapture(queue);
    virtual void beginCapture(Queue& queue) = 0;
    virtual void endCapture(Queue& queue) = 0;

    // Launch captured graph on a queue
    virtual void launch(Queue& queue) = 0;

    // Update graph topology after parameter mutations ()
    virtual void update() = 0;
};

// ============================================================================
// Enums
// ============================================================================

enum class StreamPriority { Low, Normal, High };
enum class MemoryKind { Device, HostPinned, Managed };
enum class MapMode { Read, Write, ReadWrite };

// #1475 step 2 — call-site/phase attribution for the runtime staging census.
// Push/pop a label onto a thread-local stack; the runtime's map/unmap/H2D/D2H/
// D2D/sync/submit counters attribute to the current top label, dumped per-label
// at process exit. No-ops unless NAVATALA_GPU_RUNTIME_PROFILE is set. Place
// RuntimeProfileScope at phase/call-site boundaries, e.g.:
//   GpuRuntime::RuntimeProfileScope s("pressure.gamg.halo");
void runtimeProfilePushLabel(const char* label);
void runtimeProfilePopLabel();
struct RuntimeProfileScope {
    explicit RuntimeProfileScope(const char* label) { runtimeProfilePushLabel(label); }
    ~RuntimeProfileScope() { runtimeProfilePopLabel(); }
    RuntimeProfileScope(const RuntimeProfileScope&) = delete;
    RuntimeProfileScope& operator=(const RuntimeProfileScope&) = delete;
};

// Atomic operation types for capability probing
enum class AtomicOpType {
    Add, Sub, Min, Max, And, Or, Xor, Exchange
};

// Data types for capability probing
enum class DataType {
    Float64, Float32, Float16, BFloat16,
    Int64, Int32, Int16, Int8,
    UInt64, UInt32, UInt16, UInt8,
    Complex64,   // Complex<float>
    Complex128   // Complex<double>
};

// Device version information
struct DeviceVersion {
    int major;
    int minor;
    int patch = 0;
};

// ============================================================================
// Device (Extended with metadata and graph creation)
// ============================================================================

class Device {
public:
    // Factory: create device by ID (auto-selects backend via GPU_RUNTIME_BACKEND env var)
    static std::unique_ptr<Device> create(int device_id);
    virtual ~Device() = default;

    // Queue creation
    virtual std::unique_ptr<Queue> createQueue(StreamPriority priority = StreamPriority::Normal) = 0;

    // Buffer allocation
    virtual std::unique_ptr<Buffer> createBuffer(size_t size, MemoryKind kind) = 0;

    // Program compilation/loading
    virtual std::unique_ptr<Program> createProgram(const ProgramSource& source) = 0;

    // Graph creation (CUDA/HIP only)
    virtual std::unique_ptr<Graph> createGraph() = 0;

    // Event creation (for multi-queue synchronization)
    virtual std::unique_ptr<Event> createEvent() = 0;

    // Device metadata queries
    virtual std::string getName() const = 0;
    virtual size_t getTotalMemory() const = 0;
    virtual std::string getComputeCapability() const = 0;
    virtual bool supportsFeature(const std::string& feature) const = 0;

    // Extended capability queries for runtime probing
    virtual DeviceVersion getVersion() const = 0;
    virtual bool hasExtension(const std::string& extension) const = 0;
    virtual uint32_t getSubgroupSize() const = 0;
    virtual bool supportsAtomicOp(AtomicOpType op, DataType type) const = 0;
    virtual size_t getMaxSharedMemoryPerWorkgroup() const = 0;
    virtual uint32_t getMaxWorkgroupSize() const = 0;
    virtual std::vector<std::string> getSupportedExtensions() const = 0;

    // Query current device memory usage. Returns true if the query is
    // supported by the backend (CUDA, HIP). On other backends returns false.
    virtual bool queryMemoryInfo(size_t* freeBytes, size_t* totalBytes) const {
        if (freeBytes) *freeBytes = 0;
        if (totalBytes) *totalBytes = 0;
        return false;
    }
};

// ============================================================================
// Queue (Extended with event recording and waiting)
// ============================================================================

class Queue {
public:
    virtual ~Queue() = default;

    // Kernel submission
    // `grid_*`: number of workgroups (dispatch dimensions).
    // `block_*`: threads per workgroup (workgroup size / local size).
    virtual void submit(Program& program, const std::vector<Buffer*>& args,
                        std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t grid_z,
                        std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z) = 0;

    // Memory operations
    virtual void memcpy(Buffer& dst, const Buffer& src, size_t size) = 0;
    virtual void synchronize() = 0;

    // Backend-native handle (e.g. cudaStream_t / hipStream_t / cl_command_queue / VkQueue)
    virtual void* nativeHandle() = 0;

    // Event recording
    virtual void record(Event& event) = 0;

    // Wait for event before submitting further work
    virtual void wait(Event& event) = 0;

    // Wait with timeout (microseconds), returns true if completed
    virtual bool waitFor(Event& event, uint64_t timeout_us) = 0;

    // Multi-queue synchronization (wait for event from another queue)
    virtual void wait(Event& event, Queue& other) = 0;
};

// ============================================================================
// Buffer
// ============================================================================

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t sizeBytes() const = 0;
    virtual MemoryKind memoryKind() const = 0;
    virtual void* getDevicePointer() = 0;
    virtual void* getHostPointer() = 0;
    virtual void map(MapMode mode) = 0;
    // Convenience: map then return a host pointer.
    void* mapPtr(MapMode mode = MapMode::ReadWrite) { map(mode); return getHostPointer(); }
    void* map() { return mapPtr(); }
    virtual void unmap() = 0;

    // Get backend handle (CUdeviceptr, hipDeviceptr_t, cl_mem, VkBuffer, etc.)
    // Note: handles are opaque values; constness of the returned pointer is not meaningful.
    virtual void* nativeHandle() const = 0;
};

// ============================================================================
// FFI Bridge for Runtime Capability Probing ()
// ============================================================================

extern "C" {

// Opaque device handle for FFI
typedef void* GpuRuntimeDeviceHandle;

// Version structure
struct GpuRuntimeVersion {
    int major;
    int minor;
    int patch;
};

// Create a device for capability probing
// Returns: opaque device handle, or nullptr on failure
GpuRuntimeDeviceHandle GpuRuntime_createDevice(int device_id);

// Query device capabilities
// All pointer parameters can be nullptr if that value is not needed
void GpuRuntime_queryCapabilities(
    GpuRuntimeDeviceHandle device,
    int* out_version_major,
    int* out_version_minor,
    int* out_version_patch,
    char* out_extensions,
    int extensions_buf_size,
    uint32_t* out_subgroup_size,
    size_t* out_max_shared_memory,
    uint32_t* out_max_workgroup_size,
    uint8_t* out_atomic_float64,
    uint8_t* out_atomic_float32,
    uint8_t* out_atomic_float16,
    uint8_t* out_atomic_bfloat16,
    uint8_t* out_atomic_int64,
    uint8_t* out_atomic_int32,
    uint8_t* out_atomic_int16,
    uint8_t* out_atomic_int8
);

// Destroy a device created via GpuRuntime_createDevice
void GpuRuntime_destroyDevice(GpuRuntimeDeviceHandle device);

// ============================================================================
// Memory Resource FFI (RMM Formalization)
// ============================================================================

// Opaque memory resource handle
typedef void* GpuRuntimeMemoryResourceHandle;

// Create a pool memory resource
// Returns: opaque resource handle, or nullptr on failure
GpuRuntimeMemoryResourceHandle GpuRuntime_createPoolResource(
    GpuRuntimeDeviceHandle device,
    size_t initial_size,
    size_t max_size
);

// Create a binning memory resource
GpuRuntimeMemoryResourceHandle GpuRuntime_createBinningResource(
    GpuRuntimeDeviceHandle device,
    size_t min_bin,
    size_t max_bin
);

// Allocate memory from a resource
// Returns: device pointer, or 0 on failure
void* GpuRuntime_resourceAllocate(
    GpuRuntimeMemoryResourceHandle resource,
    size_t bytes,
    size_t alignment
);

// Deallocate memory from a resource
void GpuRuntime_resourceDeallocate(
    GpuRuntimeMemoryResourceHandle resource,
    void* ptr,
    size_t bytes,
    size_t alignment
);

// Destroy a memory resource
void GpuRuntime_destroyResource(GpuRuntimeMemoryResourceHandle resource);

// Get resource statistics
// out_allocated: current bytes allocated
// out_peak: peak bytes ever allocated
// out_count: number of active allocations
void GpuRuntime_getResourceStats(
    GpuRuntimeMemoryResourceHandle resource,
    size_t* out_allocated,
    size_t* out_peak,
    size_t* out_count
);

// ============================================================================
// Stream Pool FFI (RMM Formalization)
// ============================================================================

// Opaque stream pool handle
typedef void* GpuRuntimeStreamPoolHandle;

// Opaque queue handle (for FFI)
typedef void* GpuRuntimeQueueHandle;

// Create a stream pool
GpuRuntimeStreamPoolHandle GpuRuntime_createStreamPool(
    GpuRuntimeDeviceHandle device,
    size_t initial_size
);

// Acquire a queue from the pool
// Returns: opaque queue handle
GpuRuntimeQueueHandle GpuRuntime_poolAcquireQueue(GpuRuntimeStreamPoolHandle pool);

// Release a queue back to the pool
void GpuRuntime_poolReleaseQueue(GpuRuntimeStreamPoolHandle pool, GpuRuntimeQueueHandle queue);

// Destroy a stream pool
void GpuRuntime_destroyStreamPool(GpuRuntimeStreamPoolHandle pool);

// Queue synchronization
void GpuRuntime_queueSynchronize(GpuRuntimeQueueHandle queue);

} // extern "C"

} // namespace GpuRuntime
