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

// Binning Memory Resource - Size-binned pool allocation
// Part of the RMM formalization effort
//
// Uses power-of-2 bins for efficient allocation of small to medium sizes.
// Allocations are rounded up to the nearest bin size.
//
// ## Bytes Tracking Convention
//
// This resource tracks **bin-rounded allocation sizes**:
// - Requested sizes are rounded up to the next power-of-2 bin
// - Statistics reflect binned sizes, not original requests
// - Deallocation uses stored bin size for proper cache return
//
// This maintains cache coherence and accurate memory tracking.

#pragma once

#include "memory_resource.h"
#include <memory>
#include <cstddef>

namespace GpuRuntime {

/// Size-binned pool for efficient small allocations.
///
/// Maintains separate free lists for each power-of-2 size bin.
/// Allocations are rounded up to the nearest bin size.
/// Sizes larger than max_bin_size are passed to the upstream resource.
///
/// Thread-safe: Yes (protected by internal mutex).
///
/// Example usage:
///   BinningMemoryResource binning(device, 256, 1 << 20);  // 256 B to 1 MB
///   void* ptr = binning.allocate(1000);  // Gets 1024-byte block
///   binning.deallocate(ptr, 1000);
class BinningMemoryResource : public MemoryResource {
public:
    /// Construct with device.
    /// @param device Device to allocate on
    /// @param min_bin_size Minimum bin size (default: 256)
    /// @param max_bin_size Maximum bin size (default: 1 MB)
    BinningMemoryResource(Device& device,
                          size_t min_bin_size = 256,
                          size_t max_bin_size = 1ULL << 20);

    /// Construct with custom upstream resource.
    BinningMemoryResource(MemoryResource& upstream,
                          size_t min_bin_size = 256,
                          size_t max_bin_size = 1ULL << 20);

    ~BinningMemoryResource() override;

    // Non-copyable
    BinningMemoryResource(const BinningMemoryResource&) = delete;
    BinningMemoryResource& operator=(const BinningMemoryResource&) = delete;

    // Movable
    BinningMemoryResource(BinningMemoryResource&& other) noexcept;
    BinningMemoryResource& operator=(BinningMemoryResource&& other) noexcept;

    /// Get statistics
    struct Statistics {
        /// Number of bins
        size_t num_bins;
        /// Total cached bytes across all bins
        size_t cached_bytes;
        /// Total allocations served from cache
        size_t cache_hits;
        /// Total allocations requiring upstream allocation
        size_t cache_misses;
    };

    Statistics get_statistics() const;

    /// Get base resource statistics (for polymorphic access)
    ResourceStats get_stats() const override;

    /// Get minimum bin size
    size_t min_bin_size() const;

    /// Get maximum bin size
    size_t max_bin_size() const;

    /// Get the bin size for a given allocation size
    size_t bin_size_for(size_t bytes) const;

    /// Clear all cached blocks
    void clear_cache();

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace GpuRuntime
