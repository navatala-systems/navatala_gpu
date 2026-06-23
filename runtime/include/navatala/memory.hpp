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

// Navatala C++ Wrapper Layer - Memory Management
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"

#include <cstddef>
#include <memory>
#include <atomic>
#include <mutex>

/// @file navatala/memory.hpp
/// @brief GPU memory resource abstractions and pool allocators.
///
/// This header provides:
/// - memory_resource: Abstract base for GPU memory allocators
/// - pool: RAII memory pool for efficient GPU allocations
/// - Statistics tracking for memory usage
///
/// The memory management follows a pattern similar to std::pmr::memory_resource,
/// adapted for GPU memory with proper alignment and async-safety considerations.

// Forward declare FFI types
struct NavatalaGpuMemoryResource;
struct NavatalaGpuContext;

namespace navatala {

// Forward declarations
class resources;

namespace memory {

// ============================================================================
// Memory Statistics
// ============================================================================

/// @brief Statistics for memory resource usage.
struct memory_stats {
    size_t allocated_bytes;      ///< Currently allocated bytes
    size_t peak_allocated_bytes; ///< Peak allocation during lifetime
    size_t total_allocations;    ///< Total number of allocations made
    size_t total_deallocations;  ///< Total number of deallocations made
    size_t active_allocations;   ///< Current number of active allocations
    size_t capacity_bytes;       ///< Total capacity (0 if unlimited)
    size_t free_bytes;           ///< Available bytes for allocation

    memory_stats() noexcept
        : allocated_bytes(0)
        , peak_allocated_bytes(0)
        , total_allocations(0)
        , total_deallocations(0)
        , active_allocations(0)
        , capacity_bytes(0)
        , free_bytes(0)
    {}
};

// ============================================================================
// Memory Resource Base Class
// ============================================================================

/// @brief Abstract base class for GPU memory resources.
///
/// This class provides a unified interface for GPU memory allocation,
/// similar to std::pmr::memory_resource. Implementations include:
/// - Direct device allocator (no pooling)
/// - Pool allocator (sub-allocates from large blocks)
/// - Arena allocator (fast bump allocation, bulk deallocation)
///
/// @thread_safety All methods must be thread-safe in derived implementations.
class memory_resource {
public:
    /// Default alignment for GPU allocations (typically 256 bytes for coalescing).
    static constexpr size_t default_alignment = 256;

    virtual ~memory_resource() = default;

    /// Allocates memory from the resource.
    /// @param bytes The number of bytes to allocate.
    /// @param alignment The required alignment (default: 256).
    /// @return An expected<void*> containing the allocated pointer or an error.
    [[nodiscard]] virtual result<void*> allocate(
        size_t bytes,
        size_t alignment = default_alignment) = 0;

    /// Deallocates memory previously allocated from this resource.
    /// @param ptr The pointer to deallocate.
    /// @param bytes The size of the allocation (must match allocate call).
    /// @param alignment The alignment used during allocation.
    virtual void deallocate(
        void* ptr,
        size_t bytes,
        size_t alignment = default_alignment) noexcept = 0;

    /// Returns current memory statistics.
    /// @return A memory_stats structure with current usage information.
    [[nodiscard]] virtual memory_stats get_stats() const noexcept = 0;

    /// Returns the total bytes currently allocated from this resource.
    [[nodiscard]] virtual size_t allocated_bytes() const noexcept = 0;

    /// Returns the total capacity of this resource in bytes.
    /// @return Capacity in bytes, or 0 if unlimited.
    [[nodiscard]] virtual size_t capacity_bytes() const noexcept = 0;

    /// Returns the free bytes available in this resource.
    [[nodiscard]] virtual size_t free_bytes() const noexcept = 0;

    /// Checks if this resource owns the given memory pointer.
    /// @param ptr The pointer to check.
    /// @return True if this resource owns the pointer.
    [[nodiscard]] virtual bool owns(void* ptr) const noexcept = 0;

    /// Returns the raw FFI handle. Internal use only.
    [[nodiscard]] virtual NavatalaGpuMemoryResource* handle() const noexcept = 0;

    /// Checks if this resource is valid.
    [[nodiscard]] virtual bool is_valid() const noexcept = 0;

    /// Resets the resource, releasing all allocations.
    /// @warning This invalidates all pointers previously returned by allocate().
    virtual void reset() = 0;

protected:
    memory_resource() = default;

    // Non-copyable
    memory_resource(const memory_resource&) = delete;
    memory_resource& operator=(const memory_resource&) = delete;
};

// ============================================================================
// Pool Memory Resource
// ============================================================================

/// @brief GPU memory pool for efficient sub-allocation.
///
/// A pool pre-allocates a large block of GPU memory and sub-allocates
/// from it. This reduces the overhead of individual cudaMalloc/hipMalloc
/// calls and can improve performance for workloads with many small allocations.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///   auto pool_result = pool::create(res, 1024*1024*256);  // 256 MB pool
///   if (pool_result.has_value()) {
///       auto& mem_pool = pool_result.value();
///       auto ptr = mem_pool.allocate(1024).value();
///       // Use ptr...
///       mem_pool.deallocate(ptr, 1024);
///   }
/// @endcode
///
/// @thread_safety All methods are thread-safe.
class pool : public memory_resource {
public:
    /// Creates a new memory pool.
    /// @param res The GPU resources context.
    /// @param initial_size Initial pool size in bytes (default: from config).
    /// @param max_size Maximum pool size in bytes (0 = unlimited growth).
    /// @return An expected<pool> containing the new pool or an error.
    [[nodiscard]] static result<pool> create(
        resources& res,
        size_t initial_size = NAVATALA_DEFAULT_POOL_INITIAL_SIZE,
        size_t max_size = NAVATALA_DEFAULT_POOL_MAX_SIZE);

    /// Default constructor creates an invalid pool.
    pool() noexcept;

    /// Destructor. Releases all memory back to the system.
    ~pool() noexcept override;

    // Non-copyable
    pool(const pool&) = delete;
    pool& operator=(const pool&) = delete;

    // Movable
    pool(pool&& other) noexcept;
    pool& operator=(pool&& other) noexcept;

    // memory_resource interface implementation
    [[nodiscard]] result<void*> allocate(
        size_t bytes,
        size_t alignment = default_alignment) override;

    void deallocate(
        void* ptr,
        size_t bytes,
        size_t alignment = default_alignment) noexcept override;

    [[nodiscard]] memory_stats get_stats() const noexcept override;
    [[nodiscard]] size_t allocated_bytes() const noexcept override;
    [[nodiscard]] size_t capacity_bytes() const noexcept override;
    [[nodiscard]] size_t free_bytes() const noexcept override;
    [[nodiscard]] bool owns(void* ptr) const noexcept override;
    [[nodiscard]] NavatalaGpuMemoryResource* handle() const noexcept override;
    [[nodiscard]] bool is_valid() const noexcept override;
    void reset() override;

    /// Returns the initial size configuration.
    [[nodiscard]] size_t initial_size() const noexcept { return initial_size_; }

    /// Returns the maximum size configuration.
    [[nodiscard]] size_t max_size() const noexcept { return max_size_; }

private:
    friend class navatala::resources;

    // Private constructor for factory method
    explicit pool(NavatalaGpuMemoryResource* handle,
                  NavatalaGpuContext* ctx,
                  size_t initial_size,
                  size_t max_size) noexcept;

    NavatalaGpuMemoryResource* handle_;
    NavatalaGpuContext* context_;
    size_t initial_size_;
    size_t max_size_;

    // Statistics tracking (thread-safe)
    mutable std::mutex stats_mutex_;
    mutable memory_stats stats_;
};

// ============================================================================
// Scoped Allocator
// ============================================================================

/// @brief RAII helper for temporary GPU memory allocation.
///
/// Automatically deallocates memory when the scope exits.
///
/// Example:
/// @code
///   {
///       scoped_allocation alloc(pool, 4096);
///       if (alloc.is_valid()) {
///           use_memory(alloc.data());
///       }
///   }  // Memory automatically freed here
/// @endcode
class scoped_allocation {
public:
    /// Creates a scoped allocation from a memory resource.
    /// @param mr The memory resource to allocate from.
    /// @param bytes Number of bytes to allocate.
    /// @param alignment Required alignment.
    scoped_allocation(memory_resource& mr, size_t bytes,
                      size_t alignment = memory_resource::default_alignment)
        : resource_(&mr)
        , ptr_(nullptr)
        , bytes_(bytes)
        , alignment_(alignment)
    {
        auto result = mr.allocate(bytes, alignment);
        if (result.has_value()) {
            ptr_ = result.value();
        }
    }

    /// Destructor. Deallocates the memory.
    ~scoped_allocation() noexcept {
        if (ptr_ && resource_) {
            resource_->deallocate(ptr_, bytes_, alignment_);
        }
    }

    // Non-copyable
    scoped_allocation(const scoped_allocation&) = delete;
    scoped_allocation& operator=(const scoped_allocation&) = delete;

    // Movable
    scoped_allocation(scoped_allocation&& other) noexcept
        : resource_(other.resource_)
        , ptr_(other.ptr_)
        , bytes_(other.bytes_)
        , alignment_(other.alignment_)
    {
        other.ptr_ = nullptr;
        other.resource_ = nullptr;
    }

    scoped_allocation& operator=(scoped_allocation&& other) noexcept {
        if (this != &other) {
            if (ptr_ && resource_) {
                resource_->deallocate(ptr_, bytes_, alignment_);
            }
            resource_ = other.resource_;
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            alignment_ = other.alignment_;
            other.ptr_ = nullptr;
            other.resource_ = nullptr;
        }
        return *this;
    }

    /// Returns the allocated pointer, or nullptr if allocation failed.
    [[nodiscard]] void* data() const noexcept { return ptr_; }

    /// Returns the size of the allocation in bytes.
    [[nodiscard]] size_t size() const noexcept { return bytes_; }

    /// Returns true if the allocation succeeded.
    [[nodiscard]] bool is_valid() const noexcept { return ptr_ != nullptr; }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Releases ownership of the allocation (caller takes responsibility).
    /// @return The allocated pointer.
    [[nodiscard]] void* release() noexcept {
        void* p = ptr_;
        ptr_ = nullptr;
        resource_ = nullptr;
        return p;
    }

private:
    memory_resource* resource_;
    void* ptr_;
    size_t bytes_;
    size_t alignment_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline pool::pool() noexcept
    : handle_(nullptr)
    , context_(nullptr)
    , initial_size_(0)
    , max_size_(0)
    , stats_()
{}

inline pool::pool(NavatalaGpuMemoryResource* handle,
                  NavatalaGpuContext* ctx,
                  size_t initial_size,
                  size_t max_size) noexcept
    : handle_(handle)
    , context_(ctx)
    , initial_size_(initial_size)
    , max_size_(max_size)
    , stats_()
{
    stats_.capacity_bytes = max_size > 0 ? max_size : SIZE_MAX;
    stats_.free_bytes = stats_.capacity_bytes;
}

inline pool::pool(pool&& other) noexcept
    : handle_(other.handle_)
    , context_(other.context_)
    , initial_size_(other.initial_size_)
    , max_size_(other.max_size_)
{
    std::lock_guard<std::mutex> lock(other.stats_mutex_);
    stats_ = other.stats_;
    other.handle_ = nullptr;
    other.context_ = nullptr;
}

inline pool& pool::operator=(pool&& other) noexcept {
    if (this != &other) {
        // Move construct a temp, then swap non-mutex members
        // This avoids the destructor + placement-new pattern
        pool tmp(std::move(other));

        // Swap primitive members directly
        std::swap(handle_, tmp.handle_);
        std::swap(context_, tmp.context_);
        std::swap(initial_size_, tmp.initial_size_);
        std::swap(max_size_, tmp.max_size_);

        // Swap stats under both locks (acquire in consistent order to avoid deadlock)
        std::lock_guard<std::mutex> lock_this(stats_mutex_);
        std::lock_guard<std::mutex> lock_tmp(tmp.stats_mutex_);
        std::swap(stats_, tmp.stats_);
    }
    return *this;
}

inline bool pool::is_valid() const noexcept {
    return handle_ != nullptr;
}

inline NavatalaGpuMemoryResource* pool::handle() const noexcept {
    return handle_;
}

} // namespace memory
} // namespace navatala
