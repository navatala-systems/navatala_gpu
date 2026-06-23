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

// Navatala C++ Wrapper Layer - GPU Resource Management
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "backend.hpp"
#include "queue.hpp"
#include "memory.hpp"

#include <memory>
#include <utility>
#include <vector>

/// @file navatala/resource.hpp
/// @brief GPU context and resource management.
///
/// This header provides:
/// - resources: Main GPU context manager with RAII semantics
///
/// The resources class is the root object for GPU compute operations.
/// It manages the GPU context, provides queue creation, and owns the
/// default memory pool.

// Forward declare FFI types
struct NavatalaGpuContext;

namespace navatala {

// ============================================================================
// Resources Class
// ============================================================================

/// @brief Main GPU context manager with RAII semantics.
///
/// The resources class is the entry point for all GPU operations. It:
/// - Manages the underlying GPU context lifetime
/// - Provides access to device capabilities
/// - Creates and manages execution queues
/// - Owns a default memory pool for allocations
///
/// Example:
/// @code
///   // Create with auto-selected backend
///   auto res = navatala::resources::create();
///   if (!res.has_value()) {
///       std::cerr << "Failed to create GPU context\n";
///       return 1;
///   }
///
///   // Access the resources
///   auto& gpu = res.value();
///   std::cout << "Backend: " << backend_name(gpu.get_backend()) << "\n";
///   std::cout << "Device: " << gpu.capabilities().device_name << "\n";
///
///   // Create a queue for operations
///   auto q = gpu.create_queue().value();
///   q.sync();
/// @endcode
///
/// @thread_safety The resources object itself is not thread-safe.
///                Use one per thread or synchronize access externally.
///                Queues created from resources can be used from different threads.
class resources {
public:
    /// Creates GPU resources with auto-selected or specified backend.
    /// @param b The desired backend (default: auto_select).
    /// @return An expected<resources> containing valid resources or an error.
    [[nodiscard]] static result<resources> create(backend b = backend::auto_select);

    /// Creates GPU resources for a specific device on the specified backend.
    /// @param device_id The GPU device ID (0-based).
    /// @param b The desired backend (default: auto_select).
    /// @return An expected<resources> containing valid resources or an error.
    [[nodiscard]] static result<resources> create(int device_id, backend b = backend::auto_select);

    /// Default constructor creates invalid (empty) resources.
    resources() noexcept;

    /// Destructor. Releases the GPU context and all associated resources.
    /// @note All queues, buffers, and other objects created from this
    ///       resources instance become invalid after destruction.
    ~resources() noexcept;

    // Non-copyable
    resources(const resources&) = delete;
    resources& operator=(const resources&) = delete;

    // Movable
    resources(resources&& other) noexcept;
    resources& operator=(resources&& other) noexcept;

    /// Creates a new execution queue.
    /// @param priority The queue priority (default: normal).
    /// @return An expected<queue> containing the new queue or an error.
    [[nodiscard]] result<queue> create_queue(
        queue::priority priority = queue::priority::normal);

    /// Creates a new synchronization event.
    /// @return An expected<event> containing the new event or an error.
    [[nodiscard]] result<event> create_event();

    /// Returns a reference to the default queue.
    /// @return Const reference to the default queue.
    /// @note The default queue is created during resources initialization.
    [[nodiscard]] const queue& default_queue() const noexcept { return default_queue_; }

    /// Returns a mutable reference to the default queue.
    [[nodiscard]] queue& default_queue() noexcept { return default_queue_; }

    /// Synchronizes the default queue.
    /// Convenience method equivalent to default_queue().sync().
    void sync() const { default_queue_.sync(); }

    /// Returns the device ID of this context.
    [[nodiscard]] int get_device_id() const noexcept { return device_id_; }

    /// Returns the backend type of this context.
    [[nodiscard]] backend get_backend() const noexcept { return backend_; }

    /// Returns the capabilities of the GPU device.
    [[nodiscard]] const backend_capabilities& capabilities() const noexcept {
        return capabilities_;
    }

    /// Returns a reference to the default memory pool.
    /// @return Reference to the memory pool.
    [[nodiscard]] memory::pool& get_memory_pool() noexcept { return *memory_pool_; }

    /// Returns a const reference to the default memory pool.
    [[nodiscard]] const memory::pool& get_memory_pool() const noexcept { return *memory_pool_; }

    /// Creates a new memory pool with custom configuration.
    /// @param initial_size Initial pool size in bytes.
    /// @param max_size Maximum pool size (0 = unlimited).
    /// @return An expected<memory::pool> containing the new pool or an error.
    [[nodiscard]] result<memory::pool> create_memory_pool(
        size_t initial_size = NAVATALA_DEFAULT_POOL_INITIAL_SIZE,
        size_t max_size = NAVATALA_DEFAULT_POOL_MAX_SIZE);

    /// Returns the raw FFI handle. Internal use only.
    [[nodiscard]] NavatalaGpuContext* handle() const noexcept { return handle_; }

    /// Checks if this resources object is valid.
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion for validity check.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Swap two resources objects.
    friend void swap(resources& a, resources& b) noexcept;

private:
    // Private constructor for factory methods
    explicit resources(NavatalaGpuContext* h, backend b, int device_id,
                       queue&& default_q, std::unique_ptr<memory::pool> pool);

    NavatalaGpuContext* handle_;
    backend backend_;
    int device_id_;
    backend_capabilities capabilities_;
    queue default_queue_;
    std::unique_ptr<memory::pool> memory_pool_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline resources::resources() noexcept
    : handle_(nullptr)
    , backend_(backend::auto_select)
    , device_id_(-1)
    , capabilities_()
    , default_queue_()
    , memory_pool_(nullptr)
{}

inline resources::resources(NavatalaGpuContext* h, backend b, int device_id,
                            queue&& default_q, std::unique_ptr<memory::pool> pool)
    : handle_(h)
    , backend_(b)
    , device_id_(device_id)
    , capabilities_()
    , default_queue_(std::move(default_q))
    , memory_pool_(std::move(pool))
{
    if (h) {
        capabilities_ = get_capabilities(b, device_id);
    }
}

inline resources::resources(resources&& other) noexcept
    : handle_(other.handle_)
    , backend_(other.backend_)
    , device_id_(other.device_id_)
    , capabilities_(std::move(other.capabilities_))
    , default_queue_(std::move(other.default_queue_))
    , memory_pool_(std::move(other.memory_pool_))
{
    other.handle_ = nullptr;
    other.device_id_ = -1;
}

inline resources& resources::operator=(resources&& other) noexcept {
    if (this != &other) {
        // Use swap idiom to safely handle self-assignment and exception safety
        resources tmp(std::move(other));
        swap(*this, tmp);
    }
    return *this;
}

/// @brief Swaps two resources objects.
inline void swap(resources& a, resources& b) noexcept {
    using std::swap;
    swap(a.handle_, b.handle_);
    swap(a.backend_, b.backend_);
    swap(a.device_id_, b.device_id_);
    swap(a.capabilities_, b.capabilities_);
    swap(a.default_queue_, b.default_queue_);
    swap(a.memory_pool_, b.memory_pool_);
}

} // namespace navatala
