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

// GPU Runtime Program Cache
// Persistent disk cache for compiled GPU programs

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <optional>

namespace GpuRuntime {

// Compute SHA-256 hash of source code (for cache validation)
std::string computeSourceHash(const std::string& source);

// Try to load program from disk cache
bool tryLoadFromDiskCache(
    const std::string& backend,
    const std::string& kernelName,
    const std::string& source,
    std::vector<std::uint8_t>& outBytes
);

// Save program to disk cache
bool saveToDiskCache(
    const std::string& backend,
    const std::string& kernelName,
    const std::string& source,
    const std::vector<std::uint8_t>& bytes
);

// Clear disk cache for a specific backend
bool clearDiskCache(const std::string& backend);

// Cache statistics
struct CacheStats {
    size_t fileCount;
    size_t totalBytes;
};

std::optional<CacheStats> getCacheStats(const std::string& backend);

} // namespace GpuRuntime
