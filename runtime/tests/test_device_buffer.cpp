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

// GPU Runtime - Device Buffer Tests
// Tests for DeviceBuffer<T> and DeviceScalar<T>

#include "../include/gpu_runtime.h"
#include "../include/device_buffer.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <numeric>

using namespace GpuRuntime;

// ============================================================================
// DeviceBuffer Basic Tests
// ============================================================================

void test_device_buffer_create() {
    std::cout << "=== Test: DeviceBuffer Creation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Default constructor
    DeviceBuffer<float> empty_buf;
    assert(empty_buf.empty() && "Default buffer should be empty");
    assert(empty_buf.data() == nullptr && "Default buffer data should be null");
    assert(empty_buf.size() == 0);

    // Construct with size
    DeviceBuffer<float> buf(*device, 1024);
    assert(!buf.empty() && "Buffer should not be empty");
    assert(buf.data() != nullptr && "Buffer data should not be null");
    assert(buf.size() == 1024);
    assert(buf.size_bytes() == 1024 * sizeof(float));

    std::cout << "PASS: DeviceBuffer creation" << std::endl;
}

void test_device_buffer_move() {
    std::cout << "=== Test: DeviceBuffer Move Semantics ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    DeviceBuffer<int> buf1(*device, 512);
    void* original_ptr = buf1.data();
    size_t original_size = buf1.size();

    // Move construct
    DeviceBuffer<int> buf2(std::move(buf1));
    assert(buf2.data() == original_ptr && "Move should transfer pointer");
    assert(buf2.size() == original_size && "Move should transfer size");
    assert(buf1.empty() && "Moved-from buffer should be empty");
    assert(buf1.data() == nullptr && "Moved-from buffer data should be null");

    // Move assign
    DeviceBuffer<int> buf3(*device, 256);
    buf3 = std::move(buf2);
    assert(buf3.data() == original_ptr);
    assert(buf3.size() == original_size);
    assert(buf2.empty());

    std::cout << "PASS: DeviceBuffer move semantics" << std::endl;
}

void test_device_buffer_copy_host() {
    std::cout << "=== Test: DeviceBuffer Copy From/To Host ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    assert(queue != nullptr);

    const size_t N = 256;
    std::vector<float> host_data(N);
    for (size_t i = 0; i < N; ++i) {
        host_data[i] = static_cast<float>(i) * 1.5f;
    }

    // Create device buffer and copy data to device
    DeviceBuffer<float> buf(*device, N);
    buf.copy_from_host(host_data.data(), N, *queue);
    queue->synchronize();

    // Copy data back to host
    std::vector<float> result(N);
    buf.copy_to_host(result.data(), N, *queue);

    // Verify data
    for (size_t i = 0; i < N; ++i) {
        assert(result[i] == host_data[i] && "Data should match after round-trip");
    }

    std::cout << "PASS: DeviceBuffer copy from/to host" << std::endl;
}

void test_device_buffer_partial_copy() {
    std::cout << "=== Test: DeviceBuffer Partial Copy ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    const size_t N = 256;

    std::vector<int> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 0);

    DeviceBuffer<int> buf(*device, N);

    // Copy only half
    buf.copy_from_host(host_data.data(), N / 2, *queue);
    queue->synchronize();

    // Read back only half
    std::vector<int> result(N / 2);
    buf.copy_to_host(result.data(), N / 2, *queue);

    for (size_t i = 0; i < N / 2; ++i) {
        assert(result[i] == static_cast<int>(i));
    }

    std::cout << "PASS: DeviceBuffer partial copy" << std::endl;
}

void test_device_buffer_resize() {
    std::cout << "=== Test: DeviceBuffer Resize ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    DeviceBuffer<double> buf(*device, 100);
    assert(buf.size() == 100);

    buf.resize(200);
    assert(buf.size() == 200);
    assert(buf.data() != nullptr);

    buf.resize(50);
    assert(buf.size() == 50);

    buf.resize(0);
    assert(buf.size() == 0);
    assert(buf.empty());

    std::cout << "PASS: DeviceBuffer resize" << std::endl;
}

void test_device_buffer_resize_preserve() {
    std::cout << "=== Test: DeviceBuffer Resize Preserve ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    const size_t N = 100;

    std::vector<int> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 0);

    DeviceBuffer<int> buf(*device, N);
    buf.copy_from_host(host_data.data(), N, *queue);
    queue->synchronize();

    // Resize to larger, preserving data
    buf.resize_preserve(N * 2, *queue);
    assert(buf.size() == N * 2);

    // Read back original portion
    std::vector<int> result(N);
    buf.copy_to_host(result.data(), N, *queue);

    for (size_t i = 0; i < N; ++i) {
        assert(result[i] == static_cast<int>(i) && "Data should be preserved after resize");
    }

    std::cout << "PASS: DeviceBuffer resize preserve" << std::endl;
}

void test_device_buffer_release() {
    std::cout << "=== Test: DeviceBuffer Release ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    DeviceBuffer<float> buf(*device, 100);
    assert(!buf.empty());

    // Release ownership
    auto released_buffer = buf.release();
    assert(released_buffer != nullptr && "Released buffer should be valid");
    assert(buf.empty() && "Buffer should be empty after release");
    assert(buf.data() == nullptr);

    // The released buffer should still be usable
    assert(released_buffer->getDevicePointer() != nullptr);

    std::cout << "PASS: DeviceBuffer release" << std::endl;
}

void test_device_buffer_memory_kinds() {
    std::cout << "=== Test: DeviceBuffer Memory Kinds ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Device memory
    DeviceBuffer<int> device_buf(*device, 100, MemoryKind::Device);
    assert(!device_buf.empty());
    assert(device_buf.data() != nullptr);

    // Host pinned memory
    DeviceBuffer<int> pinned_buf(*device, 100, MemoryKind::HostPinned);
    assert(!pinned_buf.empty());
    assert(pinned_buf.data() != nullptr);

    // Managed memory (may not be supported on all devices)
    try {
        DeviceBuffer<int> managed_buf(*device, 100, MemoryKind::Managed);
        assert(!managed_buf.empty());
    } catch (const std::runtime_error&) {
        // Managed memory may not be supported
    }

    std::cout << "PASS: DeviceBuffer memory kinds" << std::endl;
}

void test_queue_memcpy_explicit_map_unmap() {
    std::cout << "=== Test: Queue Memcpy Explicit Map/Unmap ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    assert(queue != nullptr);

    const size_t N = 1024;
    auto host_in = device->createBuffer(sizeof(float) * N, MemoryKind::HostPinned);
    auto host_out = device->createBuffer(sizeof(float) * N, MemoryKind::HostPinned);
    auto dev = device->createBuffer(sizeof(float) * N, MemoryKind::Device);
    assert(host_in && host_out && dev);

    {
        float* p = static_cast<float*>(host_in->mapPtr(MapMode::Write));
        assert(p != nullptr);
        for (size_t i = 0; i < N; ++i) {
            p[i] = static_cast<float>(i) * 0.5f;
        }
        host_in->unmap();
    }

    queue->memcpy(*dev, *host_in, sizeof(float) * N);
    queue->memcpy(*host_out, *dev, sizeof(float) * N);
    queue->synchronize();

    const float* out = static_cast<const float*>(host_out->mapPtr(MapMode::Read));
    assert(out != nullptr);
    for (size_t i = 0; i < N; ++i) {
        assert(out[i] == static_cast<float>(i) * 0.5f);
    }
    host_out->unmap();

    std::cout << "PASS: Queue memcpy explicit map/unmap" << std::endl;
}

// ============================================================================
// DeviceScalar Tests
// ============================================================================

void test_device_scalar_create() {
    std::cout << "=== Test: DeviceScalar Creation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Uninitialized scalar
    DeviceScalar<int> scalar(*device);
    assert(scalar.data() != nullptr);

    std::cout << "PASS: DeviceScalar creation" << std::endl;
}

void test_device_scalar_set_get() {
    std::cout << "=== Test: DeviceScalar Set/Get ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    DeviceScalar<int> scalar(*device);

    // Set value
    scalar.set(42, *queue);
    queue->synchronize();

    // Get value
    int value = scalar.get(*queue);
    assert(value == 42 && "Scalar value should be 42");

    // Set different value
    scalar.set_sync(-123, *queue);
    value = scalar.get(*queue);
    assert(value == -123 && "Scalar value should be -123");

    std::cout << "PASS: DeviceScalar set/get" << std::endl;
}

void test_device_scalar_float() {
    std::cout << "=== Test: DeviceScalar Float ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    DeviceScalar<float> scalar(*device);

    scalar.set(3.14159f, *queue);
    queue->synchronize();

    float value = scalar.get(*queue);
    assert(std::abs(value - 3.14159f) < 1e-5f && "Float value should match");

    std::cout << "PASS: DeviceScalar float" << std::endl;
}

void test_device_scalar_double() {
    std::cout << "=== Test: DeviceScalar Double ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    DeviceScalar<double> scalar(*device);

    scalar.set(2.718281828459045, *queue);
    queue->synchronize();

    double value = scalar.get(*queue);
    assert(std::abs(value - 2.718281828459045) < 1e-10 && "Double value should match");

    std::cout << "PASS: DeviceScalar double" << std::endl;
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_device_buffer_zero_size() {
    std::cout << "=== Test: DeviceBuffer Zero Size ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    DeviceBuffer<int> buf(*device, 0);
    assert(buf.empty());
    assert(buf.size() == 0);
    assert(buf.size_bytes() == 0);

    std::cout << "PASS: DeviceBuffer zero size" << std::endl;
}

void test_device_buffer_large_allocation() {
    std::cout << "=== Test: DeviceBuffer Large Allocation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Try allocating 100 MB
    try {
        DeviceBuffer<float> buf(*device, 25 * 1024 * 1024);  // 100 MB
        assert(!buf.empty());
        std::cout << "PASS: DeviceBuffer large allocation" << std::endl;
    } catch (const std::runtime_error&) {
        std::cout << "SKIP: Not enough GPU memory for large allocation" << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Runtime Device Buffer Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // DeviceBuffer tests
    test_device_buffer_create();
    test_device_buffer_move();
    test_device_buffer_copy_host();
    test_device_buffer_partial_copy();
    test_device_buffer_resize();
    test_device_buffer_resize_preserve();
    test_device_buffer_release();
    test_device_buffer_memory_kinds();
    test_queue_memcpy_explicit_map_unmap();

    // DeviceScalar tests
    test_device_scalar_create();
    test_device_scalar_set_get();
    test_device_scalar_float();
    test_device_scalar_double();

    // Edge cases
    test_device_buffer_zero_size();
    test_device_buffer_large_allocation();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Device Buffer Tests Completed!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
