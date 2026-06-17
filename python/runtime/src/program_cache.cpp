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

#include <gpu_runtime.h>
#include "program_cache.h"
#include <fstream>
#include <filesystem>
#include <optional>
#include <sstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <unistd.h>

namespace GpuRuntime {

namespace fs = std::filesystem;

// Simple SHA-256 implementation (from public domain implementation)
// This avoids external dependencies while providing proper hash security
static std::string computeSHA256(const std::string& str) {
    std::vector<uint8_t> hash(32);

    // Initial hash values
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Round constants
    const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ae, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // Pre-processing: add padding
    const uint64_t msg_len = str.size() * 8;
    std::vector<uint8_t> msg(str.begin(), str.end());
    msg.push_back(0x80);  // Append '1' bit

    // Pad with zeros to make length 56 mod 64
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00);
    }

    // Append length as 64-bit big-endian
    for (int i = 7; i >= 0; i--) {
        msg.push_back((msg_len >> (i * 8)) & 0xff);
    }

    // Process message in 512-bit (64-byte) chunks
    auto rotate_right = [](uint32_t x, uint32_t n) -> uint32_t {
        return (x >> n) | (x << (32 - n));
    };

    auto choice = [](uint32_t x, uint32_t y, uint32_t z) -> uint32_t {
        return (x & y) | (~x & z);
    };

    auto majority = [](uint32_t x, uint32_t y, uint32_t z) -> uint32_t {
        return (x & y) | (x & z) | (y & z);
    };

    auto sigma0 = [rotate_right](uint32_t x) -> uint32_t {
        return rotate_right(x, 2) ^ rotate_right(x, 13) ^ rotate_right(x, 22);
    };

    auto sigma1 = [rotate_right](uint32_t x) -> uint32_t {
        return rotate_right(x, 6) ^ rotate_right(x, 11) ^ rotate_right(x, 25);
    };

    for (size_t chunk = 0; chunk < msg.size() / 64; chunk++) {
        uint32_t w[64];

        // Copy chunk into first 16 words
        for (int i = 0; i < 16; i++) {
            w[i] = (static_cast<uint32_t>(msg[chunk * 64 + i * 4]) << 24) |
                   (static_cast<uint32_t>(msg[chunk * 64 + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[chunk * 64 + i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(msg[chunk * 64 + i * 4 + 3]));
        }

        // Extend to 48 more words
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = sigma0(w[i - 15]);
            uint32_t s1 = sigma1(w[i - 2]);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        // Initialize working variables
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h_val = h[7];

        // Compression loop
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = sigma1(e);
            uint32_t ch = choice(e, f, g);
            uint32_t temp1 = h_val + S1 + ch + k[i] + w[i];

            uint32_t S0 = sigma0(a);
            uint32_t maj = majority(a, b, c);
            uint32_t temp2 = S0 + maj;

            h_val = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        // Add compressed chunk to hash
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;
    }

    // Convert to hex string
    std::ostringstream oss;
    for (int i = 0; i < 8; i++) {
        oss << std::hex << std::setw(8) << std::setfill('0') << h[i];
    }
    return oss.str();
}

// Compute cache directory path
static std::optional<fs::path> getCacheDirectory(const std::string& backend) {
    const char* cache_home = std::getenv("XDG_CACHE_HOME");
    fs::path cache_dir;

    if (cache_home && cache_home[0] != '\0') {
        cache_dir = fs::path(cache_home) / "gpu_runtime" / backend;
    } else {
        const char* home = std::getenv("HOME");
        if (!home || home[0] == '\0') {
            return std::nullopt;  // Cannot determine cache location
        }
        cache_dir = fs::path(home) / ".cache" / "gpu_runtime" / backend;
    }

    // Create directory if it doesn't exist
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    if (ec) {
        return std::nullopt;  // Failed to create cache directory
    }

    return cache_dir;
}

// Compute SHA-256 hash of source code
std::string computeSourceHash(const std::string& source) {
    return computeSHA256(source);
}

// Get cache file path for a kernel
static std::optional<fs::path> getCacheFilePath(
    const std::string& backend,
    const std::string& kernelName,
    const std::string& source
) {
    auto cache_dir = getCacheDirectory(backend);
    if (!cache_dir) {
        return std::nullopt;
    }

    std::string hashSource = source;
    if (backend == "opencl") {
        // OpenCL compilation options may vary by environment. Include the effective "CL std"
        // selection knob in the cache key so we never load an incompatible binary when the
        // user changes GPU_RUNTIME_OPENCL_CL_STD.
        const char* stdEnv = std::getenv("GPU_RUNTIME_OPENCL_CL_STD");
        const std::string stdTag = (stdEnv && stdEnv[0] != '\0') ? std::string(stdEnv) : std::string("default(CL1.2)");
        hashSource = "opencl_cl_std=" + stdTag + "\n" + source;
    }

    std::string hash = computeSourceHash(hashSource);
    std::string extension;

    if (backend == "cuda") {
        extension = ".ptx";
    } else if (backend == "hip") {
        extension = ".hsaco";
    } else if (backend == "vulkan") {
        extension = ".spv";
    } else if (backend == "opencl") {
        extension = ".bin";
    } else if (backend == "metal") {
        extension = ".metallib";
    } else {
        return std::nullopt;  // Unknown backend
    }

    std::string filename = kernelName + "_" + hash + extension;
    return *cache_dir / filename;
}

// Try to load program from disk cache
bool tryLoadFromDiskCache(
    const std::string& backend,
    const std::string& kernelName,
    const std::string& source,
    std::vector<std::uint8_t>& outBytes
) {
    auto cache_path = getCacheFilePath(backend, kernelName, source);
    if (!cache_path) {
        return false;
    }

    // Check if file exists
    if (!fs::exists(*cache_path)) {
        return false;
    }

    // Read file
    std::ifstream file(*cache_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read into buffer
    outBytes.resize(size);
    file.read(reinterpret_cast<char*>(outBytes.data()), size);

    return file.good();
}

// Save program to disk cache
bool saveToDiskCache(
    const std::string& backend,
    const std::string& kernelName,
    const std::string& source,
    const std::vector<std::uint8_t>& bytes
) {
    auto cache_path = getCacheFilePath(backend, kernelName, source);
    if (!cache_path) {
        return false;
    }

    // Write to a unique temporary path first so parallel ranks never observe
    // a partially written cache artifact. Publish with an atomic rename.
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const fs::path tmpPath =
        cache_path->string()
      + ".tmp."
      + std::to_string(static_cast<long long>(::getpid()))
      + "."
      + std::to_string(static_cast<unsigned long long>(tid))
      + "."
      + std::to_string(static_cast<long long>(now));

    std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.flush();
    const bool ok = file.good();
    file.close();
    if (!ok) {
        std::error_code ec;
        fs::remove(tmpPath, ec);
        return false;
    }

    std::error_code ec;
    fs::rename(tmpPath, *cache_path, ec);
    if (ec) {
        std::error_code rmEc;
        fs::remove(*cache_path, rmEc);
        ec.clear();
        fs::rename(tmpPath, *cache_path, ec);
    }
    if (ec) {
        std::error_code rmEc;
        fs::remove(tmpPath, rmEc);
        return false;
    }
    return true;
}

// Clear disk cache for a specific backend
bool clearDiskCache(const std::string& backend) {
    auto cache_dir = getCacheDirectory(backend);
    if (!cache_dir) {
        return false;
    }

    std::error_code ec;
    fs::remove_all(*cache_dir, ec);
    return !ec;
}

std::optional<CacheStats> getCacheStats(const std::string& backend) {
    auto cache_dir = getCacheDirectory(backend);
    if (!cache_dir) {
        return std::nullopt;
    }

    CacheStats stats = {0, 0};

    for (const auto& entry : fs::directory_iterator(*cache_dir)) {
        if (entry.is_regular_file()) {
            stats.fileCount++;
            stats.totalBytes += entry.file_size();
        }
    }

    return stats;
}

} // namespace GpuRuntime
