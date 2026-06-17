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

// GPU Runtime - Device Selection Tests
// Test device enumeration, backend selection, and device metadata

#include "../include/gpu_runtime.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace GpuRuntime;

void test_device_factory() {
    std::cout << "=== Test: Device Factory ===" << std::endl;

    // Test creating device with ID 0
    auto device = Device::create(0);
    if (!device) {
        std::cout << "WARNING: No GPU device available - skipping device tests" << std::endl;
        return;
    }

    assert(device != nullptr && "Device creation should succeed");
    std::cout << "PASS: Device factory created device" << std::endl;
}

void test_device_metadata() {
    std::cout << "=== Test: Device Metadata ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Test device name
    std::string name = device->getName();
    assert(!name.empty() && "Device name should not be empty");
    std::cout << "Device Name: " << name << std::endl;

    // Test total memory
    size_t totalMemory = device->getTotalMemory();
    assert(totalMemory > 0 && "Device should have positive memory");
    std::cout << "Total Memory: " << (totalMemory / (1024 * 1024)) << " MB" << std::endl;

    // Test compute capability
    std::string computeCap = device->getComputeCapability();
    std::cout << "Compute Capability: " << computeCap << std::endl;

    std::cout << "PASS: Device metadata queries" << std::endl;
}

void test_device_features() {
    std::cout << "=== Test: Device Feature Support ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Test float64 support
    bool float64 = device->supportsFeature("float64");
    std::cout << "Float64 Support: " << (float64 ? "Yes" : "No") << std::endl;

    // Test float16 support
    bool float16 = device->supportsFeature("float16");
    std::cout << "Float16 Support: " << (float16 ? "Yes" : "No") << std::endl;

    // Test managed memory support
    bool managed = device->supportsFeature("managedMemory");
    std::cout << "Managed Memory: " << (managed ? "Yes" : "No") << std::endl;

    // Test shared memory support
    bool shared = device->supportsFeature("sharedMem");
    assert(shared && "All GPUs should support shared memory");
    std::cout << "Shared Memory: Yes" << std::endl;

    // Test graph support (CUDA/HIP only)
    bool graphs = device->supportsFeature("graphs");
    std::cout << "Graph Support: " << (graphs ? "Yes" : "No") << std::endl;

    std::cout << "PASS: Device feature queries" << std::endl;
}

void test_queue_creation() {
    std::cout << "=== Test: Queue Creation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Test creating queues with different priorities
    auto normalQueue = device->createQueue(StreamPriority::Normal);
    assert(normalQueue != nullptr && "Normal queue creation should succeed");
    std::cout << "Created normal queue" << std::endl;

    auto highQueue = device->createQueue(StreamPriority::High);
    assert(highQueue != nullptr && "High priority queue creation should succeed");
    std::cout << "Created high priority queue" << std::endl;

    auto lowQueue = device->createQueue(StreamPriority::Low);
    assert(lowQueue != nullptr && "Low priority queue creation should succeed");
    std::cout << "Created low priority queue" << std::endl;

    std::cout << "PASS: Queue creation" << std::endl;
}

void test_buffer_allocation() {
    std::cout << "=== Test: Buffer Allocation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Test device memory allocation
    auto deviceBuffer = device->createBuffer(1024 * 1024, MemoryKind::Device);
    assert(deviceBuffer != nullptr && "Device buffer creation should succeed");
    assert(deviceBuffer->sizeBytes() == 1024 * 1024 && "Buffer size should match");
    std::cout << "Created 1MB device buffer" << std::endl;

    // Test host-pinned memory allocation
    auto pinnedBuffer = device->createBuffer(1024 * 1024, MemoryKind::HostPinned);
    assert(pinnedBuffer != nullptr && "Pinned buffer creation should succeed");
    assert(pinnedBuffer->sizeBytes() == 1024 * 1024 && "Buffer size should match");
    assert(pinnedBuffer->getHostPointer() != nullptr && "Pinned buffer should have host pointer");
    std::cout << "Created 1MB pinned buffer" << std::endl;

    // Test managed memory allocation (if supported)
    if (device->supportsFeature("managedMemory")) {
        auto managedBuffer = device->createBuffer(1024 * 1024, MemoryKind::Managed);
        assert(managedBuffer != nullptr && "Managed buffer creation should succeed");
        assert(managedBuffer->sizeBytes() == 1024 * 1024 && "Buffer size should match");
        std::cout << "Created 1MB managed buffer" << std::endl;
    } else {
        std::cout << "SKIP: Managed memory not supported" << std::endl;
    }

    std::cout << "PASS: Buffer allocation" << std::endl;
}

void test_buffer_host_access() {
    std::cout << "=== Test: Buffer Host Access ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Test host-pinned buffer read/write
    auto pinnedBuffer = device->createBuffer(1024, MemoryKind::HostPinned);
    void* hostPtr = pinnedBuffer->getHostPointer();
    assert(hostPtr != nullptr && "Should have host pointer");

    // Write to host memory
    int* data = static_cast<int*>(hostPtr);
    const size_t numElements = 256;
    for (size_t i = 0; i < numElements; ++i) {
        data[i] = static_cast<int>(i);
    }

    // Verify write
    for (size_t i = 0; i < numElements; ++i) {
        assert(data[i] == static_cast<int>(i) && "Data should match");
    }

    std::cout << "PASS: Buffer host access" << std::endl;
}

void test_multi_device() {
    std::cout << "=== Test: Multi-Device ===" << std::endl;

    // Try to create device 0 and device 1
    auto device0 = Device::create(0);
    if (!device0) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    std::cout << "Device 0: " << device0->getName() << std::endl;

    auto device1 = Device::create(1);
    if (device1) {
        std::cout << "Device 1: " << device1->getName() << std::endl;
        std::cout << "Multiple devices available" << std::endl;
    } else {
        std::cout << "Only one device available" << std::endl;
    }

    std::cout << "PASS: Multi-device enumeration" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Runtime Device Selection Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_device_factory();
        std::cout << std::endl;

        test_device_metadata();
        std::cout << std::endl;

        test_device_features();
        std::cout << std::endl;

        test_queue_creation();
        std::cout << std::endl;

        test_buffer_allocation();
        std::cout << std::endl;

        test_buffer_host_access();
        std::cout << std::endl;

        test_multi_device();
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
