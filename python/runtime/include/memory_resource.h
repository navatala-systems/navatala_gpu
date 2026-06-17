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

// Memory Resource - Polymorphic memory allocation interface
// Part of the RMM formalization effort
//
// This provides a polymorphic interface for GPU memory allocation,
// allowing different allocation strategies (pool, binning, etc.)
// to be composed using adaptors.

#pragma once

#include <gpu_runtime.h>
#include <memory>
#include <cstddef>
#include <stdexcept>

namespace GpuRuntime {

/// Exception thrown when allocation fails
class AllocationError : public std::runtime_error {
public:
    explicit AllocationError(const std::string& what) : std::runtime_error(what) {}
    AllocationError(size_t bytes, size_t alignment)
        : std::runtime_error("Failed to allocate " + std::to_string(bytes) +
                             " bytes with alignment " + std::to_string(alignment)) {}
};

/// Abstract base for polymorphic memory allocation.
///
/// This follows the design of RMM's device_memory_resource and
/// C++17's std::pmr::memory_resource.
///
/// Derived classes implement do_allocate/do_deallocate to provide
/// actual allocation behavior.
///
/// Thread-safety: Derived classes document their thread-safety.
/// Statistics returned by memory resources
struct ResourceStats {
    size_t allocation_count = 0;      ///< Number of active allocations
    size_t allocated_bytes = 0;       ///< Currently allocated bytes
    size_t peak_allocated_bytes = 0;  ///< Peak allocated bytes
};

class MemoryResource {
public:
    virtual ~MemoryResource() = default;

    /// Get resource statistics.
    /// @return ResourceStats with allocation count, bytes, and peak
    /// @note Default implementation returns zeros. Derived classes override
    ///       to provide meaningful statistics.
    virtual ResourceStats get_stats() const { return ResourceStats{}; }

    /// Allocate memory.
    /// @param bytes Number of bytes to allocate
    /// @param alignment Size alignment hint (allocation size rounded up to multiple)
    /// @return Pointer to allocated memory
    /// @throws AllocationError on failure
    /// @note This is a SIZE alignment guarantee, not POINTER alignment.
    ///       The returned pointer alignment depends on the GPU backend (typically 256 bytes).
    ///       Actual allocated size may be larger than requested due to rounding.
    void* allocate(size_t bytes, size_t alignment = 256) {
        return do_allocate(bytes, alignment);
    }

    /// Deallocate memory.
    /// @param ptr Pointer previously returned by allocate()
    /// @param bytes Size hint (implementation may ignore - stored actual size used)
    /// @param alignment Alignment hint (implementation may ignore)
    /// @note Most implementations store actual allocation size and ignore bytes/alignment.
    ///       Callers should pass original requested values for consistency, but
    ///       implementations must not rely on caller-supplied values for correctness.
    void deallocate(void* ptr, size_t bytes, size_t alignment = 256) {
        do_deallocate(ptr, bytes, alignment);
    }

    /// Check if two resources are equivalent.
    /// Equivalent resources can deallocate each other's allocations.
    bool is_equal(const MemoryResource& other) const noexcept {
        return do_is_equal(other);
    }

    /// Equality operator
    bool operator==(const MemoryResource& other) const noexcept {
        return is_equal(other);
    }

    bool operator!=(const MemoryResource& other) const noexcept {
        return !is_equal(other);
    }

protected:
    /// Override in derived classes to perform allocation.
    virtual void* do_allocate(size_t bytes, size_t alignment) = 0;

    /// Override in derived classes to perform deallocation.
    virtual void do_deallocate(void* ptr, size_t bytes, size_t alignment) = 0;

    /// Override in derived classes to check equality.
    virtual bool do_is_equal(const MemoryResource& other) const noexcept = 0;
};

/// Non-owning reference to a memory resource.
///
/// Useful for passing memory resources without transferring ownership.
/// Similar to std::pmr::polymorphic_allocator.
class MemoryResourceRef {
public:
    /// Construct from pointer (does not take ownership)
    explicit MemoryResourceRef(MemoryResource* resource) noexcept
        : resource_(resource) {}

    /// Allocate using the referenced resource
    void* allocate(size_t bytes, size_t alignment = 256) {
        return resource_->allocate(bytes, alignment);
    }

    /// Deallocate using the referenced resource
    void deallocate(void* ptr, size_t bytes, size_t alignment = 256) {
        resource_->deallocate(ptr, bytes, alignment);
    }

    /// Get the underlying resource
    MemoryResource* resource() noexcept { return resource_; }
    const MemoryResource* resource() const noexcept { return resource_; }

private:
    MemoryResource* resource_;
};

/// Default memory resource that allocates directly from the device.
///
/// Uses Device::createBuffer() internally, so allocation patterns
/// map to cudaMalloc/hipMalloc/etc.
///
/// Thread-safe: Yes (relies on backend thread-safety).
class DefaultMemoryResource : public MemoryResource {
public:
    /// Construct a default memory resource.
    /// @param device Device to allocate on
    /// @param kind Memory kind (Device, HostPinned, Managed)
    explicit DefaultMemoryResource(Device& device,
                                   MemoryKind kind = MemoryKind::Device);

    ~DefaultMemoryResource() override;

    /// Get the device
    Device& device() noexcept { return device_; }
    const Device& device() const noexcept { return device_; }

    /// Get the memory kind
    MemoryKind kind() const noexcept { return kind_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    Device& device_;
    MemoryKind kind_;
    // Map from ptr to Buffer for proper deallocation
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Helper: Round up to next multiple of alignment
inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    return ((value + alignment - 1) / alignment) * alignment;
}

/// Helper: Check if value is power of 2
inline bool is_power_of_two(size_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

/// Get the default memory resource for a device.
/// This is a singleton that uses DefaultMemoryResource.
MemoryResource* get_default_resource(Device& device);

/// Set the default memory resource for a device.
/// @param device Device whose default resource is being updated
/// @param resource Resource to set (ownership NOT transferred)
/// @return Previous default resource
MemoryResource* set_default_resource(Device& device, MemoryResource* resource);

} // namespace GpuRuntime
