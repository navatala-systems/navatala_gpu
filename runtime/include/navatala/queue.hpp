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

// Navatala C++ Wrapper Layer - Queue and Event Abstraction
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "backend.hpp"

#include <utility>
#include <cstdint>

/// @file navatala/queue.hpp
/// @brief GPU queue (stream) and event abstractions for asynchronous execution.
///
/// This header provides:
/// - queue: RAII wrapper for GPU execution streams
/// - event: RAII wrapper for synchronization primitives
///
/// Queues represent execution streams that can run GPU operations asynchronously.
/// Events provide fine-grained synchronization between queues.

// Forward declare FFI types
struct NavatalaGpuQueue;
struct NavatalaGpuEvent;
struct NavatalaGpuContext;

namespace navatala {

// Forward declarations
class resources;
class event;

// ============================================================================
// Queue Class
// ============================================================================

/// @brief RAII wrapper for GPU execution queue/stream.
///
/// A queue represents an ordered sequence of GPU operations. Operations
/// submitted to the same queue execute in order, while operations on
/// different queues may execute concurrently.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///   auto q = res.create_queue().value();
///
///   // Submit operations to the queue
///   buffer.copy_from_host(data, q);
///
///   // Wait for completion
///   q.sync();
/// @endcode
///
/// @thread_safety Operations on the same queue must be serialized.
///                 Different queues can be used from different threads.
class queue {
public:
    /// @brief Queue priority levels.
    enum class priority : int {
        high = 0,      ///< High priority (low latency)
        normal = -1,   ///< Normal/default priority
        low = -2       ///< Low priority (background tasks)
    };

    /// Default constructor creates an invalid queue.
    queue() noexcept;

    /// Destructor. Releases the underlying GPU queue resource.
    /// @note Does NOT synchronize before destruction. Call sync() if needed.
    ~queue() noexcept;

    // Non-copyable
    queue(const queue&) = delete;
    queue& operator=(const queue&) = delete;

    // Movable
    queue(queue&& other) noexcept;
    queue& operator=(queue&& other) noexcept;

    /// Synchronizes the queue, blocking until all submitted operations complete.
    /// @throws exception (if enabled) on GPU error.
    void sync() const;

    /// Synchronizes the queue, returning status instead of throwing.
    /// @return status indicating success or error.
    [[nodiscard]] status sync_status() const noexcept;

    /// Checks if all operations submitted to the queue have completed.
    /// @return True if the queue is idle, false if operations are still pending.
    [[nodiscard]] bool is_ready() const noexcept;

    /// Returns the backend type associated with this queue.
    [[nodiscard]] backend get_backend() const noexcept;

    /// Returns the priority of this queue.
    [[nodiscard]] priority get_priority() const noexcept { return priority_; }

    /// Records an event at the current point in the queue.
    /// @param e The event to record.
    /// @throws exception (if enabled) on error.
    void record(event& e) const;

    /// Makes this queue wait for an event before executing further operations.
    /// @param e The event to wait for.
    /// @throws exception (if enabled) on error.
    void wait_for(const event& e);

    /// Returns the raw FFI handle. Internal use only.
    [[nodiscard]] NavatalaGpuQueue* handle() const noexcept { return handle_; }

    /// Returns true if this queue is valid (holds a GPU resource).
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion for validity check.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Swap two queue objects.
    friend void swap(queue& a, queue& b) noexcept;

private:
    friend class resources;
    friend class event;

    // Private constructor for resources class
    explicit queue(NavatalaGpuQueue* handle, NavatalaGpuContext* ctx,
                   backend b, priority p) noexcept;

    NavatalaGpuQueue* handle_;
    NavatalaGpuContext* context_;  // Non-owning reference to parent context
    backend backend_;
    priority priority_;
};

// ============================================================================
// Event Class
// ============================================================================

/// @brief RAII wrapper for GPU synchronization events.
///
/// Events provide a mechanism for fine-grained synchronization between
/// GPU operations, both within a single queue and across multiple queues.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///   auto q1 = res.create_queue().value();
///   auto q2 = res.create_queue().value();
///
///   // Create an event
///   auto evt = event::create(res).value();
///
///   // Submit work to q1 and record event
///   submit_kernel_to(q1);
///   q1.record(evt);
///
///   // Make q2 wait for q1's work
///   q2.wait_for(evt);
///   submit_dependent_kernel_to(q2);
/// @endcode
///
/// @thread_safety Events can be recorded and waited on from different threads,
///                but the same event should not be recorded concurrently.
class event {
public:
    /// Creates a new GPU event.
    /// @param res The GPU resources context.
    /// @return An expected<event> containing the new event or an error.
    [[nodiscard]] static result<event> create(resources& res);

    /// Default constructor creates an invalid event.
    event() noexcept;

    /// Destructor. Releases the underlying GPU event resource.
    ~event() noexcept;

    // Non-copyable
    event(const event&) = delete;
    event& operator=(const event&) = delete;

    // Movable
    event(event&& other) noexcept;
    event& operator=(event&& other) noexcept;

    /// Records the event in the specified queue.
    /// @param q The queue in which to record the event.
    /// @throws exception (if enabled) on error.
    void record(const queue& q);

    /// Records the event, returning status instead of throwing.
    /// @param q The queue in which to record the event.
    /// @return status indicating success or error.
    [[nodiscard]] status record_status(const queue& q) noexcept;

    /// Makes the specified queue wait for this event.
    /// @param q The queue that will wait.
    /// @throws exception (if enabled) on error.
    void stream_wait(queue& q) const;

    /// Makes the queue wait, returning status instead of throwing.
    /// @param q The queue that will wait.
    /// @return status indicating success or error.
    [[nodiscard]] status stream_wait_status(queue& q) const noexcept;

    /// Blocks the host thread until this event has completed.
    /// @throws exception (if enabled) on error.
    void synchronize() const;

    /// Blocks the host thread, returning status instead of throwing.
    /// @return status indicating success or error.
    [[nodiscard]] status synchronize_status() const noexcept;

    /// Checks if the event has completed without blocking.
    /// @return True if the event is complete, false if still pending.
    [[nodiscard]] bool is_complete() const noexcept;

    /// Checks if the event has been recorded.
    /// @return True if the event has been recorded at least once.
    [[nodiscard]] bool is_recorded() const noexcept { return recorded_; }

    /// Returns the raw FFI handle. Internal use only.
    [[nodiscard]] NavatalaGpuEvent* handle() const noexcept { return handle_; }

    /// Returns true if this event is valid (holds a GPU resource).
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion for validity check.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Swap two event objects.
    friend void swap(event& a, event& b) noexcept;

private:
    friend class queue;
    friend class resources;

    // Private constructor
    explicit event(NavatalaGpuEvent* handle, NavatalaGpuContext* ctx) noexcept;

    NavatalaGpuEvent* handle_;
    NavatalaGpuContext* context_;  // Non-owning reference
    bool recorded_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline queue::queue() noexcept
    : handle_(nullptr)
    , context_(nullptr)
    , backend_(backend::auto_select)
    , priority_(priority::normal)
{}

inline queue::queue(NavatalaGpuQueue* handle, NavatalaGpuContext* ctx,
                    backend b, priority p) noexcept
    : handle_(handle)
    , context_(ctx)
    , backend_(b)
    , priority_(p)
{}

inline queue::queue(queue&& other) noexcept
    : handle_(other.handle_)
    , context_(other.context_)
    , backend_(other.backend_)
    , priority_(other.priority_)
{
    other.handle_ = nullptr;
    other.context_ = nullptr;
}

inline queue& queue::operator=(queue&& other) noexcept {
    if (this != &other) {
        queue tmp(std::move(other));
        swap(*this, tmp);
    }
    return *this;
}

inline void swap(queue& a, queue& b) noexcept {
    using std::swap;
    swap(a.handle_, b.handle_);
    swap(a.context_, b.context_);
    swap(a.backend_, b.backend_);
    swap(a.priority_, b.priority_);
}

inline event::event() noexcept
    : handle_(nullptr)
    , context_(nullptr)
    , recorded_(false)
{}

inline event::event(NavatalaGpuEvent* handle, NavatalaGpuContext* ctx) noexcept
    : handle_(handle)
    , context_(ctx)
    , recorded_(false)
{}

inline event::event(event&& other) noexcept
    : handle_(other.handle_)
    , context_(other.context_)
    , recorded_(other.recorded_)
{
    other.handle_ = nullptr;
    other.context_ = nullptr;
    other.recorded_ = false;
}

inline event& event::operator=(event&& other) noexcept {
    if (this != &other) {
        event tmp(std::move(other));
        swap(*this, tmp);
    }
    return *this;
}

inline void swap(event& a, event& b) noexcept {
    using std::swap;
    swap(a.handle_, b.handle_);
    swap(a.context_, b.context_);
    swap(a.recorded_, b.recorded_);
}

} // namespace navatala
