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

// Resource Adaptors Implementation
// Part of the RMM formalization effort

#include "resource_adaptors.h"
#include <unordered_map>
#include <iostream>
#include <sstream>

namespace GpuRuntime {

// ============================================================================
// AlignedResourceAdaptor Implementation
// ============================================================================

AlignedResourceAdaptor::AlignedResourceAdaptor(MemoryResource& upstream, size_t alignment)
    : upstream_(upstream)
    , alignment_(alignment)
{
    if (!is_power_of_two(alignment)) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
}

AlignedResourceAdaptor::~AlignedResourceAdaptor() = default;

void* AlignedResourceAdaptor::do_allocate(size_t bytes, size_t alignment) {
    size_t actual_alignment = std::max(alignment, alignment_);
    return upstream_.allocate(bytes, actual_alignment);
}

void AlignedResourceAdaptor::do_deallocate(void* ptr, size_t bytes, size_t alignment) {
    size_t actual_alignment = std::max(alignment, alignment_);
    upstream_.deallocate(ptr, bytes, actual_alignment);
}

bool AlignedResourceAdaptor::do_is_equal(const MemoryResource& other) const noexcept {
    auto* other_aligned = dynamic_cast<const AlignedResourceAdaptor*>(&other);
    if (!other_aligned) return false;
    return alignment_ == other_aligned->alignment_ &&
           upstream_.is_equal(other_aligned->upstream_);
}

// ============================================================================
// TrackingResourceAdaptor Implementation
// ============================================================================

struct TrackingResourceAdaptor::Impl {
    std::mutex mutex;
    std::unordered_map<void*, size_t> allocations;
    std::atomic<size_t> alloc_count{0};
    std::atomic<size_t> total_bytes{0};
    std::atomic<size_t> peak_bytes{0};
};

TrackingResourceAdaptor::TrackingResourceAdaptor(MemoryResource& upstream)
    : upstream_(upstream)
    , impl_(std::make_unique<Impl>())
{
}

TrackingResourceAdaptor::~TrackingResourceAdaptor() {
    if (impl_->alloc_count > 0) {
        std::cerr << "[TrackingResourceAdaptor] WARNING: " << impl_->alloc_count
                  << " allocations still active at destruction\n";
    }
}

void* TrackingResourceAdaptor::do_allocate(size_t bytes, size_t alignment) {
    void* ptr = upstream_.allocate(bytes, alignment);

    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->allocations[ptr] = bytes;
    ++impl_->alloc_count;
    impl_->total_bytes += bytes;

    // Update peak (atomic max)
    size_t current = impl_->total_bytes.load();
    size_t peak = impl_->peak_bytes.load();
    while (current > peak && !impl_->peak_bytes.compare_exchange_weak(peak, current)) {
        // Retry
    }

    return ptr;
}

void TrackingResourceAdaptor::do_deallocate(void* ptr, size_t bytes, size_t alignment) {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto it = impl_->allocations.find(ptr);
        if (it != impl_->allocations.end()) {
            impl_->total_bytes -= it->second;
            --impl_->alloc_count;
            impl_->allocations.erase(it);
        }
    }

    upstream_.deallocate(ptr, bytes, alignment);
}

bool TrackingResourceAdaptor::do_is_equal(const MemoryResource& other) const noexcept {
    auto* other_tracking = dynamic_cast<const TrackingResourceAdaptor*>(&other);
    if (!other_tracking) return false;
    return upstream_.is_equal(other_tracking->upstream_);
}

size_t TrackingResourceAdaptor::allocation_count() const {
    return impl_->alloc_count.load();
}

size_t TrackingResourceAdaptor::total_allocated_bytes() const {
    return impl_->total_bytes.load();
}

size_t TrackingResourceAdaptor::peak_allocated_bytes() const {
    return impl_->peak_bytes.load();
}

ResourceStats TrackingResourceAdaptor::get_stats() const {
    ResourceStats result;
    result.allocation_count = impl_->alloc_count.load();
    result.allocated_bytes = impl_->total_bytes.load();
    result.peak_allocated_bytes = impl_->peak_bytes.load();
    return result;
}

void TrackingResourceAdaptor::log_allocations() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::ostringstream oss;
    oss << "[TrackingResourceAdaptor] Active allocations: " << impl_->allocations.size() << "\n";
    for (const auto& [ptr, size] : impl_->allocations) {
        oss << "  " << ptr << ": " << size << " bytes\n";
    }
    std::cerr << oss.str();
}

// ============================================================================
// LimitingResourceAdaptor Implementation
// ============================================================================

LimitingResourceAdaptor::LimitingResourceAdaptor(MemoryResource& upstream, size_t limit)
    : upstream_(upstream)
    , limit_(limit)
    , allocated_(0)
{
}

LimitingResourceAdaptor::~LimitingResourceAdaptor() = default;

void* LimitingResourceAdaptor::do_allocate(size_t bytes, size_t alignment) {
    // Atomic check-and-add
    size_t current = allocated_.load();
    while (true) {
        if (current + bytes > limit_) {
            throw AllocationError("Allocation would exceed limit of " +
                                  std::to_string(limit_) + " bytes");
        }
        if (allocated_.compare_exchange_weak(current, current + bytes)) {
            break;
        }
        // Retry with updated current
    }

    try {
        return upstream_.allocate(bytes, alignment);
    } catch (...) {
        allocated_ -= bytes;  // Roll back
        throw;
    }
}

void LimitingResourceAdaptor::do_deallocate(void* ptr, size_t bytes, size_t alignment) {
    upstream_.deallocate(ptr, bytes, alignment);
    allocated_ -= bytes;
}

bool LimitingResourceAdaptor::do_is_equal(const MemoryResource& other) const noexcept {
    auto* other_limiting = dynamic_cast<const LimitingResourceAdaptor*>(&other);
    if (!other_limiting) return false;
    return limit_ == other_limiting->limit_ &&
           upstream_.is_equal(other_limiting->upstream_);
}

size_t LimitingResourceAdaptor::remaining() const noexcept {
    size_t current = allocated_.load();
    return (current < limit_) ? (limit_ - current) : 0;
}

// ============================================================================
// StatisticsResourceAdaptor Implementation
// ============================================================================

StatisticsResourceAdaptor::StatisticsResourceAdaptor(MemoryResource& upstream)
    : upstream_(upstream)
{
}

StatisticsResourceAdaptor::~StatisticsResourceAdaptor() = default;

void* StatisticsResourceAdaptor::do_allocate(size_t bytes, size_t alignment) {
    void* ptr = upstream_.allocate(bytes, alignment);

    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.total_allocations;
    stats_.bytes_allocated += bytes;
    stats_.current_bytes += bytes;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);

    return ptr;
}

void StatisticsResourceAdaptor::do_deallocate(void* ptr, size_t bytes, size_t alignment) {
    upstream_.deallocate(ptr, bytes, alignment);

    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.total_deallocations;
    stats_.bytes_deallocated += bytes;
    stats_.current_bytes -= bytes;
}

bool StatisticsResourceAdaptor::do_is_equal(const MemoryResource& other) const noexcept {
    auto* other_stats = dynamic_cast<const StatisticsResourceAdaptor*>(&other);
    if (!other_stats) return false;
    return upstream_.is_equal(other_stats->upstream_);
}

StatisticsResourceAdaptor::DetailedStats StatisticsResourceAdaptor::get_detailed_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

ResourceStats StatisticsResourceAdaptor::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ResourceStats result;
    result.allocation_count = stats_.total_allocations - stats_.total_deallocations;
    result.allocated_bytes = stats_.current_bytes;
    result.peak_allocated_bytes = stats_.peak_bytes;
    return result;
}

void StatisticsResourceAdaptor::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t current = stats_.current_bytes;
    stats_ = DetailedStats{};
    stats_.current_bytes = current;
    stats_.peak_bytes = current;
}

} // namespace GpuRuntime
