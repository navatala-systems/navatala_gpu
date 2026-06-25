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

// GPU Runtime - Metal Backend Implementation
// /6: Metal backend with Device, Queue, Buffer, Program, Event
// Intended to be API-compatible with the Vulkan/OpenCL backends in gpu_runtime.

#include <gpu_runtime.h>

#include "device_mem_telemetry_internal.h"

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <TargetConditionals.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace GpuRuntime {

namespace {

static const char* programSourceKindName(ProgramSource::Kind kind) {
    switch (kind) {
        case ProgramSource::Kind::OpenClC: return "OpenClC";
        case ProgramSource::Kind::OpenClBinary: return "OpenClBinary";
        case ProgramSource::Kind::Spirv: return "Spirv";
        case ProgramSource::Kind::Glsl: return "Glsl";
        case ProgramSource::Kind::Ptx: return "Ptx";
        case ProgramSource::Kind::Cubin: return "Cubin";
        case ProgramSource::Kind::CudaCpp: return "CudaCpp";
        case ProgramSource::Kind::HipCpp: return "HipCpp";
        case ProgramSource::Kind::Hsaco: return "Hsaco";
        case ProgramSource::Kind::Msl: return "Msl";
        case ProgramSource::Kind::Metallib: return "Metallib";
    }
    return "Unknown";
}

static id<MTLDevice> selectMetalDevice(int device_id) {
    if (!@available(macOS 10.11, iOS 8.0, *)) {
        return nil;
    }

#if TARGET_OS_OSX
    if (@available(macOS 10.11, *)) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices && devices.count > 0) {
            NSInteger idx = device_id;
            if (idx < 0 || idx >= (NSInteger)devices.count) idx = 0;
            return devices[idx];
        }
    }
#endif

    return MTLCreateSystemDefaultDevice();
}

static MTLResourceOptions bufferOptionsFor(MemoryKind kind) {
    switch (kind) {
        case MemoryKind::Device:
            if (const char* p = std::getenv("NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS")) {
                const bool disabled = p[0] == '\0'
                    || (p[0] == '0' && p[1] == '\0')
                    || std::strcmp(p, "false") == 0
                    || std::strcmp(p, "FALSE") == 0
                    || std::strcmp(p, "off") == 0
                    || std::strcmp(p, "OFF") == 0;
                if (!disabled) {
                    return MTLResourceStorageModePrivate;
                }
            }
            // Apple GPUs use unified memory. Keeping runtime "Device" buffers
            // host-visible matches the generic Buffer API, where CFD-style
            // callers initialize device buffers via host-side staging.
            return MTLResourceStorageModeShared;
        case MemoryKind::HostPinned:
            // iOS: Shared; macOS: Shared is fine for our runtime needs (host-visible buffers).
            return MTLResourceStorageModeShared;
        case MemoryKind::Managed:
            // Use shared storage on Apple unified memory. It avoids explicit
            // managed-memory synchronization and keeps host/device copies simple.
            return MTLResourceStorageModeShared;
    }
    return MTLResourceStorageModeShared;
}

static uint32_t clampToU32(NSUInteger v) {
    if (v > static_cast<NSUInteger>(std::numeric_limits<uint32_t>::max())) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(v);
}

// ============================================================================
// #1475 runtime staging/readback census (env-gated, zero behavior change)
// ----------------------------------------------------------------------------
// Now that the checkHostGuard_ byte-scan (#1474) is gone, the next host-orch
// win is likely a small set of high-frequency host<->device transfers / scalar
// readbacks. These counters (relaxed atomic increments behind a cached-bool
// branch) census the runtime staging paths; cumulative per-process (per-rank)
// totals are dumped at exit. Enable with NAVATALA_GPU_RUNTIME_PROFILE=1.
// ============================================================================
struct RuntimeProfileCounters {
    std::atomic<std::uint64_t> hostPinnedMaps{0};
    std::atomic<std::uint64_t> hostPinnedUnmaps{0};
    std::atomic<std::uint64_t> h2dCopies{0};
    std::atomic<std::uint64_t> h2dBytes{0};
    std::atomic<std::uint64_t> d2hCopies{0};
    std::atomic<std::uint64_t> d2hBytes{0};
    std::atomic<std::uint64_t> d2dCopies{0};
    std::atomic<std::uint64_t> d2dBytes{0};
    std::atomic<std::uint64_t> hostHostCopies{0};
    std::atomic<std::uint64_t> hostHostBytes{0};
    std::atomic<std::uint64_t> syncs{0};
    std::atomic<std::uint64_t> submits{0};
    std::atomic<std::uint64_t> commandBuffers{0};
    std::atomic<std::uint64_t> computeEncoders{0};
    std::atomic<std::uint64_t> blitEncoders{0};
    std::atomic<std::uint64_t> hostVisibleCopies{0};
    std::atomic<std::uint64_t> batchDispatches{0};
    std::atomic<std::uint64_t> batchFlushes{0};
    std::atomic<std::uint64_t> batchLimitFlushes{0};
    std::atomic<std::uint64_t> skippedEmptySyncs{0};
};
inline bool runtimeProfileEnabled() {
    static const bool v = []() {
        const char* p = std::getenv("NAVATALA_GPU_RUNTIME_PROFILE");
        return p != nullptr && std::string(p) != "0";
    }();
    return v;
}
// #1475 step 2 — per-label (call-site/phase) attribution. A shared label->counters
// map (mutex-guarded on insert), a thread-local current-label stack pushed/popped by
// GpuRuntime::RuntimeProfileScope, and a thread-local cached pointer to the current
// label's counters (refreshed only on push/pop, so the hot increment path stays a
// lock-free atomic). Default label "(root)" when no scope is active.
// Intentionally-leaked process-lifetime singletons: the exit-time
// RuntimeProfileDumper locks/reads these, so they must outlive all static
// destruction (they are first-constructed during the run, i.e. AFTER the
// namespace-static dumper, so a normal local static would be destroyed before
// the dumper runs -> "mutex lock failed"). Leaking is fine for a
// process-lifetime diagnostic; the OS reclaims at exit.
inline std::map<std::string, RuntimeProfileCounters>& rtLabelMap() {
    static std::map<std::string, RuntimeProfileCounters>* m =
        new std::map<std::string, RuntimeProfileCounters>();
    return *m;
}
inline std::mutex& rtLabelMutex() {
    static std::mutex* mu = new std::mutex();
    return *mu;
}
inline std::vector<std::string>& rtLabelStack() {
    static thread_local std::vector<std::string> s;
    return s;
}
inline RuntimeProfileCounters*& rtCurrentPtr() {
    static thread_local RuntimeProfileCounters* p = nullptr;
    return p;
}
inline void rtRefreshCurrent() {
    const std::string lbl =
        rtLabelStack().empty() ? std::string("(root)") : rtLabelStack().back();
    std::lock_guard<std::mutex> lk(rtLabelMutex());
    rtCurrentPtr() = &rtLabelMap()[lbl];
}
inline RuntimeProfileCounters& runtimeProfile() {
    if (rtCurrentPtr() == nullptr) rtRefreshCurrent();
    return *rtCurrentPtr();
}
inline void rtProfileBump(std::atomic<std::uint64_t>& c, std::uint64_t n = 1) {
    c.fetch_add(n, std::memory_order_relaxed);
}
inline bool envFlagEnabled(const char* name, bool defaultValue = false) {
    const char* p = std::getenv(name);
    if (p == nullptr) return defaultValue;
    if (p[0] == '\0') return defaultValue;
    if ((p[0] == '0' && p[1] == '\0')
        || std::strcmp(p, "false") == 0
        || std::strcmp(p, "FALSE") == 0
        || std::strcmp(p, "off") == 0
        || std::strcmp(p, "OFF") == 0) {
        return false;
    }
    return true;
}
inline std::uint64_t envU64(const char* name, std::uint64_t defaultValue) {
    const char* p = std::getenv(name);
    if (p == nullptr || p[0] == '\0') return defaultValue;
    char* end = nullptr;
    unsigned long long v = std::strtoull(p, &end, 10);
    if (end == p || v == 0) return defaultValue;
    return static_cast<std::uint64_t>(v);
}
struct RuntimeProfileDumper {
    ~RuntimeProfileDumper() {
        if (!runtimeProfileEnabled()) return;
        std::lock_guard<std::mutex> lk(rtLabelMutex());
        for (const auto& kv : rtLabelMap()) {
            const RuntimeProfileCounters& c = kv.second;
            std::fprintf(stderr,
                "NAVATALA_GPU_RUNTIME_PROFILE: pid=%d label=%s"
                " hostpinned_map=%llu hostpinned_unmap=%llu"
                " h2d_count=%llu h2d_bytes=%llu"
                " d2h_count=%llu d2h_bytes=%llu"
                " d2d_count=%llu d2d_bytes=%llu"
                " hosthost_count=%llu hosthost_bytes=%llu"
                " sync=%llu submit=%llu"
                " command_buffer=%llu compute_encoder=%llu blit_encoder=%llu"
                " host_visible_copy=%llu"
                " batch_dispatch=%llu batch_flush=%llu batch_limit_flush=%llu"
                " skipped_empty_sync=%llu\n",
                static_cast<int>(::getpid()), kv.first.c_str(),
                static_cast<unsigned long long>(c.hostPinnedMaps.load()),
                static_cast<unsigned long long>(c.hostPinnedUnmaps.load()),
                static_cast<unsigned long long>(c.h2dCopies.load()),
                static_cast<unsigned long long>(c.h2dBytes.load()),
                static_cast<unsigned long long>(c.d2hCopies.load()),
                static_cast<unsigned long long>(c.d2hBytes.load()),
                static_cast<unsigned long long>(c.d2dCopies.load()),
                static_cast<unsigned long long>(c.d2dBytes.load()),
                static_cast<unsigned long long>(c.hostHostCopies.load()),
                static_cast<unsigned long long>(c.hostHostBytes.load()),
                static_cast<unsigned long long>(c.syncs.load()),
                static_cast<unsigned long long>(c.submits.load()),
                static_cast<unsigned long long>(c.commandBuffers.load()),
                static_cast<unsigned long long>(c.computeEncoders.load()),
                static_cast<unsigned long long>(c.blitEncoders.load()),
                static_cast<unsigned long long>(c.hostVisibleCopies.load()),
                static_cast<unsigned long long>(c.batchDispatches.load()),
                static_cast<unsigned long long>(c.batchFlushes.load()),
                static_cast<unsigned long long>(c.batchLimitFlushes.load()),
                static_cast<unsigned long long>(c.skippedEmptySyncs.load()));
        }
    }
};
static RuntimeProfileDumper g_runtimeProfileDumper;

}  // namespace

// Solver-facing scope push/pop (declared in gpu_runtime.h, used by
// GpuRuntime::RuntimeProfileScope). MUST live in namespace GpuRuntime (NOT the
// anonymous namespace above) so the symbol is EXPORTED and resolvable by the
// host CFD core's RuntimeProfileScope -- otherwise the cross-dylib call
// is unresolved and segfaults. The anon-namespace helpers (runtimeProfileEnabled,
// rtLabelStack, rtRefreshCurrent) remain visible here via enclosing-namespace
// lookup. No-op when profiling is off.
void runtimeProfilePushLabel(const char* label) {
    if (!runtimeProfileEnabled()) return;
    rtLabelStack().push_back(label ? label : "(null)");
    rtRefreshCurrent();
}
void runtimeProfilePopLabel() {
    if (!runtimeProfileEnabled()) return;
    auto& s = rtLabelStack();
    if (!s.empty()) s.pop_back();
    rtRefreshCurrent();
}

// ============================================================================
// MetalEvent
// ============================================================================

// Host-side poll cadence for MetalEvent::wait(). 1 ms is fast enough that
// the perf cost is well below 1 % of pressure-solve wall (event-bound
// host idle is typically sub-ms once the GPU is done), slow enough that
// polling overhead is negligible vs cv-only blocking.
inline constexpr auto kMetalEventPollInterval = std::chrono::milliseconds(1);

// Synchronization state for a MetalEvent, owned via shared_ptr so an in-flight
// Metal completion handler keeps it alive even if the MetalEvent itself is
// destroyed first.
//
// Bug fix (#314, 2026-06-01): under heavy multi-rank (np>1) GPU load, wait()
// returns as soon as the HARDWARE sharedEvent_.signaledValue reaches the target
// (isComplete()'s fast path), often BEFORE the delayed GCD completion handler
// fires. The solver then proceeds and destroys/reuses the MetalEvent. When the
// late handler finally ran `ev->markComplete()` on the freed event, it called
// cv_.notify_all() on a dead condition_variable -> os_unfair_lock_unowned_abort
// -> SIGKILL (EXC_BREAKPOINT) of a rank, tearing down the whole MPI job. Moving
// the cv/mutex/flags into a shared_ptr the completion block co-owns removes the
// use-after-free; notifying under the mutex also closes the notify-vs-destroy race.
struct MetalEventSync {
    std::atomic<bool> complete_{false};
    std::atomic<uint64_t> recordedValue_{0};
    std::mutex mu_;
    std::condition_variable cv_;

    // Versioned mark (Bug 5, 2026-05-28): only mark complete if the value the
    // GCD handler was registered against still matches the current
    // recordedValue_ — so a delayed handler from a PREVIOUS record() cannot
    // falsely complete a re-recorded event. The lock is held across the flag
    // store AND notify so a waking waiter cannot drop its shared_ptr (and
    // potentially destroy *this) while notify_all() is still in flight (#314).
    void markComplete(uint64_t expectedValue) {
        std::lock_guard<std::mutex> lk(mu_);
        if (recordedValue_.load(std::memory_order_acquire) == expectedValue) {
            complete_.store(true, std::memory_order_release);
            cv_.notify_all();
        }
    }
};

class MetalEvent final : public Event {
public:
    explicit MetalEvent(id<MTLDevice> device)
        : device_(device),
          sync_(std::make_shared<MetalEventSync>()),
          nextValue_(1) {
        if (@available(macOS 10.14, iOS 12.0, *)) {
            sharedEvent_ = [device newSharedEvent];
        } else {
            sharedEvent_ = nil;
        }
    }

    ~MetalEvent() override = default;

    bool isComplete() const override {
        if (sync_->complete_.load(std::memory_order_acquire)) {
            return true;
        }
        if (@available(macOS 10.14, iOS 12.0, *)) {
            if (sharedEvent_ != nil) {
                uint64_t want = sync_->recordedValue_.load(std::memory_order_acquire);
                if (want == 0) return false;
                return sharedEvent_.signaledValue >= want;
            }
        }
        return false;
    }

    void wait() override {
        // Metal completed handlers can be delayed under heavy multi-rank GPU
        // load. Poll the hardware shared-event timeline so host waits do not
        // depend solely on GCD callback delivery (isComplete() inspects both
        // the complete_ flag set by the handler AND sharedEvent_.signaledValue).
        std::unique_lock<std::mutex> lock(sync_->mu_);
        while (!isComplete()) {
            sync_->cv_.wait_for(lock, kMetalEventPollInterval);
        }
    }

    bool waitFor(uint64_t timeout_us) {
        std::unique_lock<std::mutex> lock(sync_->mu_);
        return sync_->cv_.wait_for(lock, std::chrono::microseconds(timeout_us), [&] { return isComplete(); });
    }

    void* nativeHandle() override {
        if (@available(macOS 10.14, iOS 12.0, *)) {
            return (__bridge void*)sharedEvent_;
        }
        return nullptr;
    }

    uint64_t recordValue() {
        sync_->complete_.store(false, std::memory_order_release);
        uint64_t v = nextValue_.fetch_add(1, std::memory_order_acq_rel);
        sync_->recordedValue_.store(v, std::memory_order_release);
        return v;
    }

    uint64_t recordedValue() const {
        uint64_t v = sync_->recordedValue_.load(std::memory_order_acquire);
        return v == 0 ? 1 : v;
    }

    id<MTLSharedEvent> sharedEvent() const {
        if (@available(macOS 10.14, iOS 12.0, *)) {
            return sharedEvent_;
        }
        return nil;
    }

    // Shared sync state — captured (by shared_ptr copy) into the GCD completion
    // block in MetalQueue::record() so the block stays valid even if this
    // MetalEvent is destroyed before the (delayed) handler fires (#314).
    const std::shared_ptr<MetalEventSync>& sync() const { return sync_; }

private:
    __strong id<MTLDevice> device_;
    __strong id<MTLSharedEvent> sharedEvent_;
    std::shared_ptr<MetalEventSync> sync_;
    std::atomic<uint64_t> nextValue_;
};

// ============================================================================
// MetalBuffer
// ============================================================================

class MetalBuffer final : public Buffer {
public:
    MetalBuffer(id<MTLDevice> device, size_t size, MemoryKind kind)
        : device_(device), kind_(kind), size_(size) {
        if (kind == MemoryKind::HostPinned) {
            hostStorage_.resize(size + 2 * kHostGuardBytes);
            std::fill(hostStorage_.begin(), hostStorage_.end(), kHostGuardByte);
            contents_ = hostStorage_.data() + kHostGuardBytes;
            storageMode_ = MTLStorageModeShared;
            if (std::getenv("GPU_RUNTIME_DEBUG_METAL_BUFFER") != nullptr) {
                fprintf(stderr,
                        "MetalBuffer create-host this=%p device=%p kind=%d size=%zu buffer=%p contents=%p metalContents=%p\n",
                        static_cast<void*>(this),
                        (__bridge void*)device,
                        static_cast<int>(kind),
                        size,
                        nullptr,
                        contents_,
                        nullptr);
            }
            return;
        }

        MTLResourceOptions options = bufferOptionsFor(kind);
        buffer_ = [device newBufferWithLength:size options:options];
        if (!buffer_) {
            throw std::runtime_error("Failed to create Metal buffer");
        }
        metalContents_ = buffer_.contents;
        contents_ = metalContents_;
        storageMode_ = buffer_.storageMode;
        noteDeviceMemAlloc(size_);
        countedBytes_ = size_;
        if (std::getenv("GPU_RUNTIME_DEBUG_METAL_BUFFER") != nullptr) {
            fprintf(stderr,
                    "MetalBuffer create this=%p device=%p kind=%d size=%zu buffer=%p contents=%p storageMode=%lu\n",
                    static_cast<void*>(this),
                    (__bridge void*)device,
                    static_cast<int>(kind),
                    size,
                    (__bridge void*)buffer_,
                    contents_,
                    static_cast<unsigned long>(storageMode_));
        }
    }

    ~MetalBuffer() override {
        if (countedBytes_ != 0) {
            noteDeviceMemFree(countedBytes_);
        }
    }

    size_t sizeBytes() const override { return size_; }
    MemoryKind memoryKind() const override { return kind_; }

    void* getDevicePointer() override {
        // Metal has no raw device pointer; use the opaque buffer handle.
        return (__bridge void*)buffer_;
    }

    void* getHostPointer() override {
        if (storageMode_ == MTLStorageModePrivate) return nullptr;
        return contents_;
    }

    void map(MapMode mode) override {
        lastMapMode_ = mode;
        if (runtimeProfileEnabled() && kind_ == MemoryKind::HostPinned) {
            rtProfileBump(runtimeProfile().hostPinnedMaps);
        }
        if (std::getenv("GPU_RUNTIME_DEBUG_METAL_BUFFER") != nullptr) {
            fprintf(stderr,
                    "MetalBuffer map this=%p kind=%d size=%zu buffer=%p\n",
                    static_cast<void*>(this),
                    static_cast<int>(kind_),
                    size_,
                    (__bridge void*)buffer_);
        }
        if (storageMode_ == MTLStorageModePrivate) {
            throw std::runtime_error("Cannot map Metal private buffer");
        }
        if (kind_ == MemoryKind::HostPinned
            && (mode == MapMode::Read || mode == MapMode::ReadWrite)) {
            checkHostGuard_("map");
            if (metalContents_ != nullptr) {
                std::memcpy(contents_, metalContents_, size_);
            }
        }
    }

    void unmap() override {
        if (runtimeProfileEnabled() && kind_ == MemoryKind::HostPinned) {
            rtProfileBump(runtimeProfile().hostPinnedUnmaps);
        }
        if (kind_ == MemoryKind::HostPinned) {
            checkHostGuard_("unmap");
            if (metalContents_ != nullptr
                && (lastMapMode_ == MapMode::Write || lastMapMode_ == MapMode::ReadWrite)) {
                std::memcpy(metalContents_, contents_, size_);
            }
            return;
        }

        // For managed buffers on macOS, inform Metal that the CPU modified the range.
#if TARGET_OS_OSX
        __unsafe_unretained id<MTLBuffer> buffer = this->buffer();
        if (storageMode_ == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(0, size_)];
        }
#endif
    }

    void* nativeHandle() const override {
        return (__bridge void*)buffer_;
    }

    id<MTLBuffer> buffer() const { return buffer_; }
    id<MTLBuffer> bufferForKernel() {
        if (buffer_ == nil) {
            buffer_ = [device_ newBufferWithLength:size_ options:MTLResourceStorageModeShared];
            if (!buffer_) {
                throw std::runtime_error("Failed to create Metal host-pinned kernel backing buffer");
            }
            metalContents_ = buffer_.contents;
            metalNoteAlloc(size_);
            countedBytes_ = size_;
            if (kind_ == MemoryKind::HostPinned && contents_ != nullptr && metalContents_ != nullptr) {
                checkHostGuard_("kernel-backing");
                std::memcpy(metalContents_, contents_, size_);
            }
        }
        return buffer_;
    }
    const void* hostPointerForCopy() const {
        return storageMode_ == MTLStorageModePrivate ? nullptr : contents_;
    }
    void* hostPointerForCopy() {
        return storageMode_ == MTLStorageModePrivate ? nullptr : contents_;
    }

private:
    static constexpr std::size_t kHostGuardBytes = 4096;
    static constexpr std::uint8_t kHostGuardByte = 0xA5u;

    void checkHostGuard_(const char* where) const {
        if (kind_ != MemoryKind::HostPinned || hostStorage_.empty()) {
            return;
        }
        // #1474 perf: the guard regions are constant 0xA5, so compare with
        // SIMD-optimized std::memcmp against a cached reference block rather
        // than a byte-by-byte std::all_of(+lambda) that did not vectorize.
        // A host-CPU `sample` of the np>1 GAMG V-cycle showed this scan was
        // ~56% of active host time (called on every HostPinned map/unmap of
        // the staging buffer). Identical guard semantics: detects ANY
        // corruption in either kHostGuardBytes region; ~10-50x cheaper.
        static const std::array<std::uint8_t, kHostGuardBytes> guardRef = [] {
            std::array<std::uint8_t, kHostGuardBytes> v{};
            v.fill(kHostGuardByte);
            return v;
        }();
        const std::uint8_t* base = hostStorage_.data();
        const bool frontOk =
            std::memcmp(base, guardRef.data(), kHostGuardBytes) == 0;
        const bool backOk =
            std::memcmp(base + kHostGuardBytes + size_, guardRef.data(),
                        kHostGuardBytes) == 0;
        if (!frontOk || !backOk) {
            throw std::runtime_error(
                std::string("Metal HostPinned buffer guard corrupted during ")
                + where
                + " size=" + std::to_string(size_));
        }
    }

    __strong id<MTLDevice> device_;
    MemoryKind kind_;
    size_t size_;
    __strong id<MTLBuffer> buffer_;
    void* contents_ = nullptr;
    void* metalContents_ = nullptr;
    MTLStorageMode storageMode_ = MTLStorageModeShared;
    MapMode lastMapMode_ = MapMode::ReadWrite;
    std::vector<std::uint8_t> hostStorage_;
    std::size_t countedBytes_ = 0;  // bytes registered in the Shared-buffer telemetry
};

// ============================================================================
// MetalProgram
// ============================================================================

// ----------------------------------------------------------------------------
// Bug 7 (Metal FSCache flock contention) mitigation helpers.
// MTLBinaryArchive lets us persist a compiled compute-pipeline binary in an
// app-managed file and recreate the pipeline from it without re-entering the
// shader compiler's shared FSCache (which serializes via flock() and wedges
// multi-rank startup). Metal only (de)serializes archives via file URLs, so
// these helpers round-trip the archive bytes through a temp file.
// ----------------------------------------------------------------------------
static bool metalBinaryArchiveEnabled() {
    const char* e = std::getenv("GPU_RUNTIME_METAL_BINARY_ARCHIVE");
    // Default ON; explicit "0" disables (falls back to direct JIT pipeline build).
    if (e && e[0] == '0' && e[1] == '\0') return false;
    return true;
}

static NSString* metalUniqueTempPath(const char* ext) {
    NSString* uniq = [[NSProcessInfo processInfo] globallyUniqueString];
    NSString* name = [NSString stringWithFormat:@"gpu_runtime_%@.%s", uniq, ext];
    return [NSTemporaryDirectory() stringByAppendingPathComponent:name];
}

static NSString* metalWriteTempFile(const std::uint8_t* data, std::size_t n,
                                    const char* ext) {
    NSString* path = metalUniqueTempPath(ext);
    NSData* d = [NSData dataWithBytes:data length:n];
    NSError* werr = nil;
    if (![d writeToFile:path options:NSDataWritingAtomic error:&werr]) return nil;
    return path;
}

static std::vector<std::uint8_t> metalReadAllBytes(NSString* path) {
    NSData* d = [NSData dataWithContentsOfFile:path];
    if (!d) return {};
    const std::uint8_t* p = static_cast<const std::uint8_t*>(d.bytes);
    return std::vector<std::uint8_t>(p, p + d.length);
}

static void metalRemoveFile(NSString* path) {
    if (path) [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
}

class MetalProgram final : public Program {
public:
    MetalProgram(id<MTLDevice> device, const ProgramSource& source)
        : device_(device), kind_(source.kind), entryPoint_(source.entryPoint) {
        if (source.kind != ProgramSource::Kind::Msl && source.kind != ProgramSource::Kind::Metallib) {
            throw std::runtime_error(
                std::string("Metal backend requires ProgramSource::Kind::Msl or Metallib")
              + " (got kind=" + programSourceKindName(source.kind)
              + ", entry=" + source.entryPoint + ")");
        }

        // Disk-cache integration: only treat Metallib inputs as "compiled bytes".
        // When source is MSL, Metal does not provide a portable, stable bytecode blob at runtime.
        // Returning empty here prevents program_cache from writing misleading ".metallib" cache entries.
        if (source.kind == ProgramSource::Kind::Metallib) {
            compiledBytes_ = source.bytes;
            compiledFormat_ = "metallib";
        } else {
            compiledBytes_.clear();
            compiledFormat_ = "msl";
        }

        NSError* error = nil;
        id<MTLLibrary> library = nil;
        if (source.kind == ProgramSource::Kind::Metallib) {
            void* owned = std::malloc(source.bytes.size());
            if (!owned) {
                throw std::runtime_error("Failed to allocate metallib buffer");
            }
            std::memcpy(owned, source.bytes.data(), source.bytes.size());
            dispatch_data_t dd = dispatch_data_create(
                owned,
                source.bytes.size(),
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^{
                    std::free(owned);
                }
            );
            library = [device newLibraryWithData:dd error:&error];
        } else {
            NSString* src = [[NSString alloc] initWithBytes:source.bytes.data()
                                                     length:source.bytes.size()
                                                   encoding:NSUTF8StringEncoding];
            if (!src) {
                throw std::runtime_error("Failed to decode MSL source as UTF-8");
            }
            MTLCompileOptions* options = [MTLCompileOptions new];
            library = [device newLibraryWithSource:src options:options error:&error];
        }

        if (!library) {
            std::string msg = error ? std::string([[error localizedDescription] UTF8String]) : "Unknown error";
            throw std::runtime_error("Failed to compile Metal library: " + msg);
        }

        NSString* kernelName = [NSString stringWithUTF8String:source.entryPoint.c_str()];
        id<MTLFunction> fn = [library newFunctionWithName:kernelName];
        if (!fn) {
            throw std::runtime_error("Failed to locate Metal kernel entry point: " + source.entryPoint);
        }

        // ---- Fast path: load the compute pipeline from a precompiled binary
        // archive (no JIT, no FSCache flock). Requires the MTLFunction above
        // (the archive stores the pipeline keyed by function, not the source).
        bool builtFromArchive = false;
        if (metalBinaryArchiveEnabled() && !source.binaryArchive.empty()) {
            if (@available(macOS 11.0, *)) {
                @try {
                    NSString* arPath = metalWriteTempFile(
                        source.binaryArchive.data(), source.binaryArchive.size(), "metalar");
                    if (arPath) {
                        MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
                        ad.url = [NSURL fileURLWithPath:arPath];
                        NSError* aerr = nil;
                        id<MTLBinaryArchive> archive =
                            [device newBinaryArchiveWithDescriptor:ad error:&aerr];
                        if (archive) {
                            MTLComputePipelineDescriptor* pd =
                                [MTLComputePipelineDescriptor new];
                            pd.computeFunction = fn;
                            pd.binaryArchives = @[archive];
                            // FailOnBinaryArchiveMiss (macOS 13+) guarantees we never
                            // silently fall back into the FSCache-flock JIT path.
                            MTLPipelineOption opt = MTLPipelineOptionNone;
                            if (@available(macOS 13.0, *)) {
                                opt = MTLPipelineOptionFailOnBinaryArchiveMiss;
                            }
                            NSError* perr = nil;
                            pipeline_ = [device
                                newComputePipelineStateWithDescriptor:pd
                                                              options:opt
                                                           reflection:nil
                                                                error:&perr];
                            if (pipeline_) builtFromArchive = true;
                        }
                        metalRemoveFile(arPath);
                    }
                } @catch (...) {
                    // Stale/incompatible archive (e.g. GPU or OS change): fall back to JIT.
                    pipeline_ = nil;
                    builtFromArchive = false;
                }
            }
        }

        // ---- Cold path: JIT-compile the pipeline. This is the FSCache flock
        // site. During multi-rank startup it is reached only on a cache miss,
        // which the rank-0 prewarm serializes (one rank builds the archives,
        // peers wait at a barrier, then all load from the archive above).
        if (!builtFromArchive) {
            error = nil;
            pipeline_ = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pipeline_) {
                std::string msg = error ? std::string([[error localizedDescription] UTF8String]) : "Unknown error";
                throw std::runtime_error("Failed to create Metal compute pipeline: " + msg);
            }
            // Best-effort: serialize a single-pipeline binary archive and expose it
            // via getCompiledBytes() so the disk cache can persist it for next time.
            // Failures here never fail the run (the pipeline is already built).
            if (metalBinaryArchiveEnabled() && source.kind == ProgramSource::Kind::Msl) {
                if (@available(macOS 11.0, *)) {
                    @try {
                        MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
                        ad.url = nil;  // empty archive to populate
                        NSError* aerr = nil;
                        id<MTLBinaryArchive> archive =
                            [device newBinaryArchiveWithDescriptor:ad error:&aerr];
                        MTLComputePipelineDescriptor* pd =
                            [MTLComputePipelineDescriptor new];
                        pd.computeFunction = fn;
                        if (archive &&
                            [archive addComputePipelineFunctionsWithDescriptor:pd error:&aerr]) {
                            NSString* outPath = metalUniqueTempPath("metalar");
                            NSURL* outUrl = [NSURL fileURLWithPath:outPath];
                            if ([archive serializeToURL:outUrl error:&aerr]) {
                                compiledBytes_ = metalReadAllBytes(outPath);
                                if (!compiledBytes_.empty()) {
                                    compiledFormat_ = "metalarchive";
                                }
                            }
                            metalRemoveFile(outPath);
                        }
                    } @catch (...) {
                        compiledBytes_.clear();
                        compiledFormat_ = "msl";
                    }
                }
            }
        }
    }

    ~MetalProgram() override = default;

    void* nativeHandle() override { return (__bridge void*)pipeline_; }

    std::vector<std::uint8_t> getCompiledBytes() const override {
        return compiledBytes_;
    }

    std::string getCompiledFormat() const override {
        return compiledFormat_;
    }

    id<MTLComputePipelineState> pipeline() const { return pipeline_; }
    const std::string& entryPoint() const { return entryPoint_; }

private:
    __strong id<MTLDevice> device_;
    ProgramSource::Kind kind_;
    std::string entryPoint_;
    std::vector<std::uint8_t> compiledBytes_;
    std::string compiledFormat_;
    __strong id<MTLComputePipelineState> pipeline_;
};

// ============================================================================
// MetalQueue
// ============================================================================

class MetalQueue final : public Queue {
public:
    MetalQueue(id<MTLDevice> device, StreamPriority priority)
        : device_(device),
          batchComputeSubmits_(envFlagEnabled("NAVATALA_GPU_METAL_BATCH_SUBMITS", false)),
          batchSubmitLimit_(envU64("NAVATALA_GPU_METAL_BATCH_LIMIT", 64)) {
        (void)priority;
        queue_ = [device newCommandQueue];
        if (!queue_) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
    }

    ~MetalQueue() override {
        try {
            (void)flushPendingCompute(false, "destructor");
        } catch (...) {
            // Destructors must not throw. Normal callers flush via synchronize,
            // memcpy, record/wait, or event synchronization before destruction.
        }
    }

    void submit(Program& program, const std::vector<Buffer*>& args,
                std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t grid_z,
                std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z) override {
        if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().submits);
        auto* mp = dynamic_cast<MetalProgram*>(&program);
        if (!mp) {
            throw std::runtime_error("MetalQueue received non-Metal program");
        }

        // #312: wrap per-dispatch command encoding in an autorelease pool.
        // [queue_ commandBuffer] / [cmd computeCommandEncoder] return
        // AUTORELEASED objects; the solver's main loop is plain C++ with no
        // run-loop to drain the thread's pool, so without an explicit pool the
        // autoreleased Metal command objects (AGX compute context, NSData arg
        // buffers, command-buffer content) accumulate for the entire run and
        // are freed only at process exit — the dominant per-step host-memory
        // growth on long Metal runs (worse at np>1 from extra halo dispatches).
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = nil;
            id<MTLComputeCommandEncoder> enc = nil;
            if (batchComputeSubmits_) {
                ensurePendingComputeEncoder();
                cmd = pendingComputeCommandBuffer_;
                enc = pendingComputeEncoder_;
                if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().batchDispatches);
            } else {
                cmd = newCommandBuffer("submit");
                enc = newComputeEncoder(cmd, "submit");
            }
            if (!cmd || !enc) {
                throw std::runtime_error("Failed to create Metal command buffer/encoder");
            }

            [enc setComputePipelineState:mp->pipeline()];
            for (size_t i = 0; i < args.size(); ++i) {
                auto* mb = dynamic_cast<MetalBuffer*>(args[i]);
                if (!mb) {
                    const char* argState = args[i] ? "non-null/non-metal" : "null";
                    throw std::runtime_error(
                        "MetalQueue arg is not a MetalBuffer: entry="
                        + mp->entryPoint()
                        + " argIndex=" + std::to_string(i)
                        + " argState=" + argState);
                }
                [enc setBuffer:mb->bufferForKernel() offset:0 atIndex:i];
            }

            // Runtime contract: grid_* are number of workgroups, block_* are threads/workgroup.
            MTLSize tg = MTLSizeMake(block_x, block_y, block_z);
            MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);

            NSUInteger threadsPerTG = tg.width * tg.height * tg.depth;
            NSUInteger maxThreads = mp->pipeline().maxTotalThreadsPerThreadgroup;
            if (threadsPerTG > maxThreads) {
                throw std::runtime_error("Metal threadgroup size exceeds device limit");
            }

            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            if (batchComputeSubmits_) {
                ++pendingComputeDispatches_;
                if (pendingComputeDispatches_ >= batchSubmitLimit_) {
                    if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().batchLimitFlushes);
                    (void)flushPendingCompute(false, "batch-limit");
                }
            } else {
                [enc endEncoding];
                [cmd commit];
                hasCommittedWork_ = true;
            }
        }
    }

    void memcpy(Buffer& dst, const Buffer& src, size_t size) override {
        memcpyOffset(dst, 0, src, 0, size);
    }

    void memcpyOffset(Buffer& dst, size_t dstOffset,
                      const Buffer& src, size_t srcOffset,
                      size_t size) override {
        auto* d = dynamic_cast<MetalBuffer*>(&dst);
        auto* s = dynamic_cast<const MetalBuffer*>(&src);
        if (!d || !s) {
            throw std::runtime_error("MetalQueue memcpyOffset requires MetalBuffer");
        }
        if (dstOffset > dst.sizeBytes() || size > dst.sizeBytes() - dstOffset ||
            srcOffset > src.sizeBytes() || size > src.sizeBytes() - srcOffset) {
            throw std::runtime_error("Metal memcpyOffset out of bounds");
        }

        (void)flushPendingCompute(false, "memcpy");

        const bool srcDev = (src.memoryKind() == MemoryKind::Device
                          || src.memoryKind() == MemoryKind::Managed);
        const bool dstDev = (dst.memoryKind() == MemoryKind::Device
                          || dst.memoryKind() == MemoryKind::Managed);

        if (runtimeProfileEnabled()) {
            // Classify by semantic direction (HostPinned staging is neither
            // Device nor Managed, so staging->field is H2D, field->staging D2H).
            auto& c = runtimeProfile();
            if (srcDev && !dstDev) {
                rtProfileBump(c.d2hCopies); rtProfileBump(c.d2hBytes, size);
            } else if (!srcDev && dstDev) {
                rtProfileBump(c.h2dCopies); rtProfileBump(c.h2dBytes, size);
            } else if (srcDev && dstDev) {
                rtProfileBump(c.d2dCopies); rtProfileBump(c.d2dBytes, size);
            } else {
                rtProfileBump(c.hostHostCopies); rtProfileBump(c.hostHostBytes, size);
            }
        }

        // Device-to-device copies should remain in the Metal command stream.
        // Metal Device/Managed buffers are usually CPU-visible on Apple UMA, so
        // a naïve host-pointer shortcut would otherwise force a queue sync and
        // do the copy on the CPU. A blit keeps ordering async like CUDA/HIP.
        if (srcDev && dstDev) {
            @autoreleasepool {  // #312: drain autoreleased command buffer/encoder per blit
                id<MTLCommandBuffer> cmd = newCommandBuffer("memcpy-d2d");
                id<MTLBlitCommandEncoder> enc = newBlitEncoder(cmd, "memcpy-d2d");
                if (!cmd || !enc) {
                    throw std::runtime_error("Failed to create Metal D2D blit command encoder");
                }

                id<MTLBuffer> srcBuffer = const_cast<MetalBuffer*>(s)->bufferForKernel();
                id<MTLBuffer> dstBuffer = d->bufferForKernel();
                [enc copyFromBuffer:srcBuffer sourceOffset:srcOffset toBuffer:dstBuffer destinationOffset:dstOffset size:size];
                [enc endEncoding];
                [cmd commit];
                hasCommittedWork_ = true;
            }
            return;
        }

        void* dstHost = d->hostPointerForCopy();
        const void* srcHost = s->hostPointerForCopy();
        if (dstHost != nullptr && srcHost != nullptr) {
            // CPU-visible shared-memory copies are outside the Metal command
            // stream. Drain prior GPU work first so device-to-host copies read
            // completed kernel results.
            if (srcDev) {
                synchronize();
            }
            if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().hostVisibleCopies);
            std::memcpy(static_cast<char*>(dstHost) + dstOffset,
                        static_cast<const char*>(srcHost) + srcOffset,
                        size);
            return;
        }

        @autoreleasepool {  // #312: drain autoreleased command buffer/encoder per blit
            id<MTLCommandBuffer> cmd = newCommandBuffer("memcpy");
            id<MTLBlitCommandEncoder> enc = newBlitEncoder(cmd, "memcpy");
            if (!cmd || !enc) {
                throw std::runtime_error("Failed to create Metal blit command encoder");
            }

            id<MTLBuffer> srcBuffer = const_cast<MetalBuffer*>(s)->bufferForKernel();
            id<MTLBuffer> dstBuffer = d->bufferForKernel();
            [enc copyFromBuffer:srcBuffer sourceOffset:srcOffset toBuffer:dstBuffer destinationOffset:dstOffset size:size];
            [enc endEncoding];
            [cmd commit];
            hasCommittedWork_ = true;
        }
    }

    void synchronize() override {
        if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().syncs);
        if (flushPendingCompute(true, "synchronize")) {
            hasCommittedWork_ = false;
            return;
        }
        if (!hasCommittedWork_) {
            if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().skippedEmptySyncs);
            return;
        }
        @autoreleasepool {  // #312: drain autoreleased command buffer per sync
            id<MTLCommandBuffer> cmd = newCommandBuffer("synchronize");
            if (!cmd) {
                throw std::runtime_error("Failed to create Metal command buffer for synchronize");
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            hasCommittedWork_ = false;
        }
    }

    void* nativeHandle() override {
        (void)flushPendingCompute(false, "nativeHandle");
        hasCommittedWork_ = true;
        return (__bridge void*)queue_;
    }

    void record(Event& event) override {
        auto* ev = dynamic_cast<MetalEvent*>(&event);
        if (!ev) throw std::runtime_error("MetalQueue record requires MetalEvent");

        (void)flushPendingCompute(false, "record");
        @autoreleasepool {  // #312: drain autoreleased command buffer per record
            id<MTLCommandBuffer> cmd = newCommandBuffer("record");
            if (!cmd) throw std::runtime_error("Failed to create Metal command buffer for record");

            uint64_t v = ev->recordValue();
            if (@available(macOS 10.14, iOS 12.0, *)) {
                if (auto se = ev->sharedEvent()) {
                    [cmd encodeSignalEvent:se value:v];
                }
            }
            // Bug 5 + #314 fix: capture the per-record value v AND a shared_ptr
            // copy of the event's sync state. v guards against a delayed handler
            // from an earlier record() falsely completing a re-recorded event;
            // the shared_ptr keeps the cv/mutex alive so a handler that fires
            // AFTER the MetalEvent is destroyed (wait() returned via the
            // hardware sharedEvent) never dereferences freed memory.
            auto sync = ev->sync();
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
                sync->markComplete(v);
            }];
            [cmd commit];
            hasCommittedWork_ = true;
        }
    }

    void wait(Event& event) override {
        auto* ev = dynamic_cast<MetalEvent*>(&event);
        if (!ev) throw std::runtime_error("MetalQueue wait requires MetalEvent");

        (void)flushPendingCompute(false, "wait");
        @autoreleasepool {  // #312: drain autoreleased command buffer per wait
            id<MTLCommandBuffer> cmd = newCommandBuffer("wait");
            if (!cmd) throw std::runtime_error("Failed to create Metal command buffer for wait");

            uint64_t v = ev->recordedValue();
            if (@available(macOS 10.14, iOS 12.0, *)) {
                if (auto se = ev->sharedEvent()) {
                    [cmd encodeWaitForEvent:se value:v];
                }
            } else {
                // Fallback: host-side wait (not GPU-synchronized).
                ev->wait();
            }
            [cmd commit];
            hasCommittedWork_ = true;
        }
    }

    bool waitFor(Event& event, uint64_t timeout_us) override {
        auto* ev = dynamic_cast<MetalEvent*>(&event);
        if (!ev) throw std::runtime_error("MetalQueue waitFor requires MetalEvent");

        if (ev->isComplete()) return true;

        return ev->waitFor(timeout_us);
    }

    void wait(Event& event, Queue& other) override {
        (void)other;
        wait(event);
    }

private:
    id<MTLCommandBuffer> newCommandBuffer(const char* where) {
        (void)where;
        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().commandBuffers);
        return cmd;
    }

    id<MTLComputeCommandEncoder> newComputeEncoder(id<MTLCommandBuffer> cmd, const char* where) {
        (void)where;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().computeEncoders);
        return enc;
    }

    id<MTLBlitCommandEncoder> newBlitEncoder(id<MTLCommandBuffer> cmd, const char* where) {
        (void)where;
        id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
        if (runtimeProfileEnabled()) rtProfileBump(runtimeProfile().blitEncoders);
        return enc;
    }

    void ensurePendingComputeEncoder() {
        if (pendingComputeCommandBuffer_ != nil && pendingComputeEncoder_ != nil) {
            return;
        }
        pendingComputeCommandBuffer_ = newCommandBuffer("batch-submit");
        pendingComputeEncoder_ = newComputeEncoder(pendingComputeCommandBuffer_, "batch-submit");
        if (!pendingComputeCommandBuffer_ || !pendingComputeEncoder_) {
            pendingComputeCommandBuffer_ = nil;
            pendingComputeEncoder_ = nil;
            pendingComputeDispatches_ = 0;
            throw std::runtime_error("Failed to create batched Metal command buffer/encoder");
        }
    }

    bool flushPendingCompute(bool waitUntilComplete, const char* where) {
        (void)where;
        if (pendingComputeCommandBuffer_ == nil) {
            return false;
        }
        id<MTLCommandBuffer> cmd = pendingComputeCommandBuffer_;
        id<MTLComputeCommandEncoder> enc = pendingComputeEncoder_;
        const std::uint64_t dispatches = pendingComputeDispatches_;

        pendingComputeCommandBuffer_ = nil;
        pendingComputeEncoder_ = nil;
        pendingComputeDispatches_ = 0;

        if (enc != nil) {
            [enc endEncoding];
        }
        [cmd commit];
        hasCommittedWork_ = true;
        if (waitUntilComplete) {
            [cmd waitUntilCompleted];
            hasCommittedWork_ = false;
        }
        if (runtimeProfileEnabled() && dispatches > 0) {
            rtProfileBump(runtimeProfile().batchFlushes);
        }
        return true;
    }

    __strong id<MTLDevice> device_;
    __strong id<MTLCommandQueue> queue_;
    __strong id<MTLCommandBuffer> pendingComputeCommandBuffer_;
    __strong id<MTLComputeCommandEncoder> pendingComputeEncoder_;
    std::uint64_t pendingComputeDispatches_ = 0;
    bool batchComputeSubmits_ = false;
    std::uint64_t batchSubmitLimit_ = 64;
    bool hasCommittedWork_ = false;
};

// ============================================================================
// MetalDevice
// ============================================================================

class MetalDevice final : public Device {
public:
    explicit MetalDevice(id<MTLDevice> device) : device_(device) {
        if (!device_) {
            throw std::runtime_error("MetalDevice constructed with null device");
        }
    }

    ~MetalDevice() override = default;

    std::unique_ptr<Queue> createQueue(StreamPriority priority = StreamPriority::Normal) override {
        return std::make_unique<MetalQueue>(device_, priority);
    }

    std::unique_ptr<Buffer> createBuffer(size_t size, MemoryKind kind) override {
        return std::make_unique<MetalBuffer>(device_, size, kind);
    }

    std::unique_ptr<Program> createProgram(const ProgramSource& source) override {
        return std::make_unique<MetalProgram>(device_, source);
    }

    std::unique_ptr<Graph> createGraph() override {
        throw std::runtime_error("Metal backend does not support graphs");
    }

    std::unique_ptr<Event> createEvent() override {
        return std::make_unique<MetalEvent>(device_);
    }

    std::string getName() const override {
        return std::string([device_.name UTF8String]);
    }

    size_t getTotalMemory() const override {
        if (@available(macOS 10.13, iOS 11.0, *)) {
            return static_cast<size_t>(device_.recommendedMaxWorkingSetSize);
        }
        return 1024ULL * 1024ULL * 1024ULL;
    }

    std::string getComputeCapability() const override {
        // Expose a compact family identifier; used only as informational.
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if ([device_ supportsFamily:MTLGPUFamilyApple7]) return "Apple7";
            if ([device_ supportsFamily:MTLGPUFamilyApple6]) return "Apple6";
            if ([device_ supportsFamily:MTLGPUFamilyApple5]) return "Apple5";
            if ([device_ supportsFamily:MTLGPUFamilyApple4]) return "Apple4";
            if ([device_ supportsFamily:MTLGPUFamilyApple3]) return "Apple3";
            if ([device_ supportsFamily:MTLGPUFamilyApple2]) return "Apple2";
            if ([device_ supportsFamily:MTLGPUFamilyApple1]) return "Apple1";
            if ([device_ supportsFamily:MTLGPUFamilyMac2]) return "Mac2";
        }
        return "Metal";
    }

    bool supportsFeature(const std::string& feature) const override {
        if (feature == "metal") return true;
        if (feature == "graphs") return false;
        if (feature == "float64" || feature == "fp64") return false;
        if (feature == "managedMemory") {
#if TARGET_OS_OSX
            return true;
#else
            return false;
#endif
        }
        if (feature == "sharedMem") return true;
        if (feature == "privateDeviceBuffers") return true;
        if (feature == "float16" || feature == "fp16") {
            if (@available(macOS 10.13, iOS 11.0, *)) return true;
            return false;
        }
        if (feature == "bfloat16" || feature == "bf16") {
            if (@available(macOS 13.0, iOS 16.0, *)) return true;
            return false;
        }
        if (feature == "streams") return true;
        if (feature == "events") return true;
        if (feature == "atomics") return true;
        if (feature.rfind("subgroup", 0) == 0) return true;
        if (feature == "tensor_cores") return false;
        if (feature == "memory_pools") return true;
        if (feature == "nurbs.rational.evaluate") return false;
        if (feature == "nurbs.rational.derivatives") return false;
        if (feature == "nurbs.rational.project") return false;
        return false;
    }

    DeviceVersion getVersion() const override {
        if (@available(macOS 11.0, iOS 14.0, *)) return DeviceVersion{3, 0, 0};
        if (@available(macOS 10.15, iOS 13.0, *)) return DeviceVersion{2, 3, 0};
        if (@available(macOS 10.13, iOS 11.0, *)) return DeviceVersion{2, 0, 0};
        return DeviceVersion{1, 0, 0};
    }

    bool hasExtension(const std::string& extension) const override {
        return extension == "metal";
    }

    uint32_t getSubgroupSize() const override {
        return 32;
    }

    bool supportsAtomicOp(AtomicOpType op, DataType type) const override {
        (void)op;
        switch (type) {
            case DataType::Float64: return false;
            case DataType::Float32:
                if (@available(macOS 11.0, iOS 14.0, *)) return true;
                return false;
            case DataType::Float16: return false;
            case DataType::BFloat16: return false;
            case DataType::Int32:
            case DataType::UInt32: return true;
            case DataType::Int64:
            case DataType::UInt64:
                if (@available(macOS 10.15, iOS 13.0, *)) return true;
                return false;
            default: return false;
        }
    }

    size_t getMaxSharedMemoryPerWorkgroup() const override {
        if (@available(macOS 10.13, iOS 11.0, *)) {
            return static_cast<size_t>(device_.maxThreadgroupMemoryLength);
        }
        return 32 * 1024;
    }

    uint32_t getMaxWorkgroupSize() const override {
        MTLSize m = device_.maxThreadsPerThreadgroup;
        uint64_t total = static_cast<uint64_t>(m.width) * static_cast<uint64_t>(m.height) * static_cast<uint64_t>(m.depth);
        if (total == 0) total = 1024;
        if (total > std::numeric_limits<uint32_t>::max()) total = std::numeric_limits<uint32_t>::max();
        return static_cast<uint32_t>(total);
    }

    std::vector<std::string> getSupportedExtensions() const override {
        return {"metal"};
    }

private:
    __strong id<MTLDevice> device_;
};

std::unique_ptr<Device> MetalDevice_tryCreate(int device_id) {
    id<MTLDevice> selected = selectMetalDevice(device_id);
    if (!selected) return nullptr;

    try {
        return std::make_unique<MetalDevice>(selected);
    } catch (...) {
        return nullptr;
    }
}

}  // namespace GpuRuntime
