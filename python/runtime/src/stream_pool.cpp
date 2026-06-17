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

// Stream Pool Implementation
// Part of the RMM formalization effort

#include "stream_pool.h"
#include <algorithm>

namespace GpuRuntime {

StreamPool::StreamPool(Device& device, size_t initial_size, StreamPriority priority)
    : device_(device)
    , default_priority_(priority)
    , total_created_(0)
{
    // Pre-create initial queues
    available_queues_.reserve(initial_size);
    for (size_t i = 0; i < initial_size; ++i) {
        available_queues_.push_back(device_.createQueue(priority));
        ++total_created_;
    }
}

StreamPool::~StreamPool() {
    // All queues are automatically destroyed by unique_ptr
}

StreamPool::StreamPool(StreamPool&& other) noexcept
    : device_(other.device_)
    , default_priority_(other.default_priority_)
    , available_queues_(std::move(other.available_queues_))
    , total_created_(other.total_created_)
{
    other.total_created_ = 0;
}

StreamPool& StreamPool::operator=(StreamPool&& other) {
    if (this != &other) {
        // Can't rebind the device reference; cross-device move-assign is a no-op.
        if (&device_ != &other.device_) {
            return *this;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        default_priority_ = other.default_priority_;
        available_queues_ = std::move(other.available_queues_);
        total_created_ = other.total_created_;
        other.total_created_ = 0;
    }
    return *this;
}

std::unique_ptr<Queue> StreamPool::acquire(StreamPriority priority) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!available_queues_.empty()) {
        // Reuse existing queue (note: priority might not match, but that's okay)
        auto queue = std::move(available_queues_.back());
        available_queues_.pop_back();
        return queue;
    }

    // Create new queue
    ++total_created_;
    return device_.createQueue(priority);
}

void StreamPool::release(std::unique_ptr<Queue> queue) {
    if (!queue) return;

    // Synchronize before returning to pool
    queue->synchronize();

    std::lock_guard<std::mutex> lock(mutex_);
    available_queues_.push_back(std::move(queue));
}

size_t StreamPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_queues_.size();
}

size_t StreamPool::total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_created_;
}

void StreamPool::trim(size_t max_available) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (available_queues_.size() > max_available) {
        available_queues_.pop_back();
        // Note: We don't decrement total_created_ because we're tracking
        // total ever created, not current count
    }
}

void StreamPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    available_queues_.clear();
}

} // namespace GpuRuntime
