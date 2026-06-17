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

// Stream Pool - Pool of reusable queues to reduce creation overhead
// Part of the RMM formalization effort

#pragma once

#include <gpu_runtime.h>
#include <memory>
#include <vector>
#include <mutex>
#include <cstddef>

namespace GpuRuntime {

/// Pool of reusable queues to reduce queue creation overhead.
///
/// Queue creation can be expensive on some backends. This pool maintains
/// a set of pre-created queues that can be acquired and released.
///
/// Thread-safe: All methods are protected by a mutex.
///
/// Example usage:
///   StreamPool pool(device, 4);  // Pre-create 4 queues
///   auto queue = pool.acquire();
///   // Use queue...
///   pool.release(std::move(queue));
class StreamPool {
public:
    /// Create pool with initial capacity.
    /// @param device The device to create queues on
    /// @param initial_size Number of queues to pre-create (default: 4)
    /// @param priority Default priority for created queues (default: Normal)
    explicit StreamPool(Device& device,
                        size_t initial_size = 4,
                        StreamPriority priority = StreamPriority::Normal);

    /// Destructor - releases all pooled queues
    ~StreamPool();

    // Non-copyable
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    // Movable
    StreamPool(StreamPool&& other) noexcept;
    StreamPool& operator=(StreamPool&& other);

    /// Acquire a queue from the pool.
    /// If no queues are available, creates a new one.
    /// @param priority Priority override (optional, uses pool default if not specified)
    /// @return Unique pointer to queue - caller owns it until release()
    std::unique_ptr<Queue> acquire(StreamPriority priority);
    std::unique_ptr<Queue> acquire() { return acquire(default_priority_); }

    /// Release a queue back to the pool for reuse.
    /// The queue is synchronized before being returned to the pool.
    /// @param queue Queue to return to pool
    void release(std::unique_ptr<Queue> queue);

    /// Get number of currently available (not in-use) queues.
    size_t available() const;

    /// Get total number of queues created by this pool.
    size_t total() const;

    /// Get number of queues currently in use.
    size_t in_use() const { return total() - available(); }

    /// Get the device this pool creates queues for.
    Device& device() { return device_; }
    const Device& device() const { return device_; }

    /// Get the default priority for this pool.
    StreamPriority default_priority() const { return default_priority_; }

    /// Trim pool to specified size by releasing excess queues.
    /// @param max_available Maximum queues to keep available
    void trim(size_t max_available);

    /// Clear all available queues (in-use queues are not affected).
    void clear();

private:
    Device& device_;
    StreamPriority default_priority_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Queue>> available_queues_;
    size_t total_created_ = 0;
};

} // namespace GpuRuntime
