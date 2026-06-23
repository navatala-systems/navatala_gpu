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

#include <gpu_runtime.h>

#include "device_mem_telemetry_internal.h"

#include <atomic>

namespace GpuRuntime {

namespace {

std::atomic<std::size_t> g_currentBytes{0};
std::atomic<std::size_t> g_peakBytes{0};
std::atomic<std::size_t> g_count{0};
std::atomic<std::size_t> g_largestBytes{0};

void updateMax(std::atomic<std::size_t>& target, std::size_t value)
{
    std::size_t observed = target.load(std::memory_order_relaxed);
    while (value > observed
        && !target.compare_exchange_weak(
            observed,
            value,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }
}

} // namespace

void noteDeviceMemAlloc(std::size_t bytes)
{
    if (bytes == 0) {
        return;
    }

    const std::size_t current =
        g_currentBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    g_count.fetch_add(1, std::memory_order_relaxed);
    updateMax(g_peakBytes, current);
    updateMax(g_largestBytes, bytes);
}

void noteDeviceMemFree(std::size_t bytes)
{
    if (bytes == 0) {
        return;
    }

    g_currentBytes.fetch_sub(bytes, std::memory_order_relaxed);
    g_count.fetch_sub(1, std::memory_order_relaxed);
}

DeviceMemTelemetry deviceMemTelemetry()
{
    DeviceMemTelemetry telemetry;
    telemetry.currentBytes = g_currentBytes.load(std::memory_order_relaxed);
    telemetry.peakBytes = g_peakBytes.load(std::memory_order_relaxed);
    telemetry.count = g_count.load(std::memory_order_relaxed);
    telemetry.largestBytes = g_largestBytes.load(std::memory_order_relaxed);
    return telemetry;
}

} // namespace GpuRuntime
