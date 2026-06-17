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

// GPU Runtime - Program Cache Tests
// Validate persistent disk cache helpers (save/load/clear/stats)

#include "../include/program_cache.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

static std::string tempCacheDir() {
    std::string base = "/tmp/gpu_runtime_cache_test_";
    base += std::to_string(static_cast<long long>(::getpid()));
    base += "_";
    base += std::to_string(static_cast<long long>(std::rand()));
    return base;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Runtime Program Cache Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    const char* oldXdg = std::getenv("XDG_CACHE_HOME");
    std::string tmp = tempCacheDir();
    setenv("XDG_CACHE_HOME", tmp.c_str(), 1);

    try {
        const std::string backend = "opencl";
        const std::string kernel = "test_kernel";
        const std::string source = "kernel void test() { }";
        const std::vector<std::uint8_t> bytes = {1, 2, 3, 4, 5, 6, 7, 8};

        // Ensure clean start
        GpuRuntime::clearDiskCache(backend);

        // Save
        bool saved = GpuRuntime::saveToDiskCache(backend, kernel, source, bytes);
        assert(saved && "saveToDiskCache should succeed");

        // Stats
        auto stats1 = GpuRuntime::getCacheStats(backend);
        assert(stats1.has_value() && "getCacheStats should return stats");
        assert(stats1->fileCount >= 1 && "cache should contain at least one file");
        assert(stats1->totalBytes >= bytes.size() && "cache should contain at least bytes size");

        // Load
        std::vector<std::uint8_t> loaded;
        bool loadedOk = GpuRuntime::tryLoadFromDiskCache(backend, kernel, source, loaded);
        assert(loadedOk && "tryLoadFromDiskCache should succeed");
        assert(loaded == bytes && "loaded bytes should match saved bytes");

        // Clear
        bool cleared = GpuRuntime::clearDiskCache(backend);
        assert(cleared && "clearDiskCache should succeed");

        auto stats2 = GpuRuntime::getCacheStats(backend);
        assert(stats2.has_value() && "getCacheStats should return stats");
        assert(stats2->fileCount == 0 && "cache should be empty after clear");

        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        // Restore env before exiting.
        if (oldXdg) setenv("XDG_CACHE_HOME", oldXdg, 1);
        else unsetenv("XDG_CACHE_HOME");
        return 1;
    }

    // Restore env
    if (oldXdg) setenv("XDG_CACHE_HOME", oldXdg, 1);
    else unsetenv("XDG_CACHE_HOME");
    return 0;
}
