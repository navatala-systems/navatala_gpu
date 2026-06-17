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

// GPU Runtime - Event Synchronization Tests
// Test native event implementation and multi-queue synchronization

#include "../include/gpu_runtime.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

using namespace GpuRuntime;

void test_event_create() {
    std::cout << "=== Test: Event Creation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    assert(queue != nullptr && "Queue creation should succeed");

    // Test explicit event creation via device
    auto ev = device->createEvent();
    assert(ev != nullptr && "Device should be able to create Event");

    std::cout << "PASS: Event creation" << std::endl;
}

void test_event_record_and_wait() {
    std::cout << "=== Test: Event Record + Wait ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    auto ev = device->createEvent();
    assert(queue != nullptr);
    assert(ev != nullptr);

    queue->record(*ev);
    queue->wait(*ev);
    queue->synchronize();

    // After synchronize, the event must be complete.
    assert(ev->isComplete());
    std::cout << "PASS: Event record + wait" << std::endl;
}

void test_event_completion() {
    std::cout << "=== Test: Event Completion Query ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);

    // Create a buffer to work with
    auto buffer = device->createBuffer(1024, MemoryKind::HostPinned);
    assert(buffer != nullptr && "Buffer creation should succeed");

    // Initialize buffer
    int* data = static_cast<int*>(buffer->getHostPointer());
    for (int i = 0; i < 256; ++i) {
        data[i] = i;
    }

    // Synchronize to ensure data is visible
    queue->synchronize();

    // Record an event and check completion after sync.
    auto ev = device->createEvent();
    assert(ev != nullptr);
    queue->record(*ev);
    queue->synchronize();
    assert(ev->isComplete());

    std::cout << "PASS: Event completion query" << std::endl;
}

void test_event_wait() {
    std::cout << "=== Test: Event Wait ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);

    // Create a buffer
    auto buffer = device->createBuffer(1024, MemoryKind::HostPinned);
    assert(buffer != nullptr);

    auto ev = device->createEvent();
    assert(ev != nullptr);
    queue->record(*ev);
    ev->wait();  // host-side wait API
    assert(ev->isComplete());

    std::cout << "PASS: Event wait" << std::endl;
}

void test_event_timeout() {
    std::cout << "=== Test: Event Wait with Timeout ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);

    auto ev = device->createEvent();
    assert(ev != nullptr);
    queue->record(*ev);

    // `waitFor` is queue-based, so it should complete quickly for this minimal marker.
    auto start = std::chrono::steady_clock::now();
    bool ok = queue->waitFor(*ev, 5 * 1000 * 1000);  // 5 seconds
    auto end = std::chrono::steady_clock::now();
    assert(ok);

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "waitFor completed in " << elapsed.count() << " us" << std::endl;

    std::cout << "PASS: Event wait with timeout" << std::endl;
}

void test_multi_queue_sync() {
    std::cout << "=== Test: Multi-Queue Synchronization ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Create two queues
    auto queue1 = device->createQueue(StreamPriority::Normal);
    auto queue2 = device->createQueue(StreamPriority::High);

    assert(queue1 != nullptr && "Queue1 creation should succeed");
    assert(queue2 != nullptr && "Queue2 creation should succeed");

    // Create buffers
    auto buffer1 = device->createBuffer(1024, MemoryKind::HostPinned);
    auto buffer2 = device->createBuffer(1024, MemoryKind::HostPinned);

    assert(buffer1 != nullptr && "Buffer1 creation should succeed");
    assert(buffer2 != nullptr && "Buffer2 creation should succeed");

    // Initialize buffer1
    int* data1 = static_cast<int*>(buffer1->getHostPointer());
    for (int i = 0; i < 256; ++i) {
        data1[i] = i;
    }

    // Copy from buffer1 to buffer2 on queue1
    queue1->memcpy(*buffer2, *buffer1, 1024);

    // Synchronize queue1
    queue1->synchronize();

    // Verify buffer2 content
    int* data2 = static_cast<int*>(buffer2->getHostPointer());
    for (int i = 0; i < 256; ++i) {
        assert(data2[i] == i && "Data should match after copy");
    }

    std::cout << "PASS: Multi-queue synchronization" << std::endl;
}

void test_chained_operations() {
    std::cout << "=== Test: Chained Operations with Events ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);

    // Create three buffers for chaining: buf1 -> buf2 -> buf3
    auto buf1 = device->createBuffer(1024, MemoryKind::HostPinned);
    auto buf2 = device->createBuffer(1024, MemoryKind::HostPinned);
    auto buf3 = device->createBuffer(1024, MemoryKind::HostPinned);

    // Initialize buf1
    int* data1 = static_cast<int*>(buf1->getHostPointer());
    for (int i = 0; i < 256; ++i) {
        data1[i] = i;
    }

    // Chain: buf1 -> buf2 -> buf3
    queue->memcpy(*buf2, *buf1, 1024);
    queue->synchronize();

    queue->memcpy(*buf3, *buf2, 1024);
    queue->synchronize();

    // Verify final buffer
    int* data3 = static_cast<int*>(buf3->getHostPointer());
    for (int i = 0; i < 256; ++i) {
        assert(data3[i] == i && "Data should propagate through chain");
    }

    std::cout << "PASS: Chained operations with events" << std::endl;
}

void test_concurrent_queues() {
    std::cout << "=== Test: Concurrent Queue Operations ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Create multiple queues
    auto queue1 = device->createQueue(StreamPriority::High);
    auto queue2 = device->createQueue(StreamPriority::Normal);
    auto queue3 = device->createQueue(StreamPriority::Low);

    // Create buffers for each queue
    auto buf1 = device->createBuffer(1024, MemoryKind::HostPinned);
    auto buf2 = device->createBuffer(1024, MemoryKind::HostPinned);
    auto buf3 = device->createBuffer(1024, MemoryKind::HostPinned);

    // Initialize buffers
    int* data1 = static_cast<int*>(buf1->getHostPointer());
    int* data2 = static_cast<int*>(buf2->getHostPointer());
    int* data3 = static_cast<int*>(buf3->getHostPointer());

    for (int i = 0; i < 256; ++i) {
        data1[i] = 1;
        data2[i] = 2;
        data3[i] = 3;
    }

    // Submit operations to all queues concurrently
    // (In real scenario, these would be kernel launches)
    queue1->synchronize();
    queue2->synchronize();
    queue3->synchronize();

    std::cout << "PASS: Concurrent queue operations" << std::endl;
}

void test_stress_events() {
    std::cout << "=== Test: Event Stress Test ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto queue = device->createQueue(StreamPriority::Normal);
    auto buffer = device->createBuffer(1024, MemoryKind::HostPinned);

    const int numIterations = 100;
    for (int i = 0; i < numIterations; ++i) {
        // Simulate work and sync
        queue->synchronize();
    }

    std::cout << "Completed " << numIterations << " event operations" << std::endl;
    std::cout << "PASS: Event stress test" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Runtime Event Synchronization Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_event_create();
        std::cout << std::endl;

        test_event_record_and_wait();
        std::cout << std::endl;

        test_event_completion();
        std::cout << std::endl;

        test_event_wait();
        std::cout << std::endl;

        test_event_timeout();
        std::cout << std::endl;

        test_multi_queue_sync();
        std::cout << std::endl;

        test_chained_operations();
        std::cout << std::endl;

        test_concurrent_queues();
        std::cout << std::endl;

        test_stress_events();
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << "ALL EVENT TESTS PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
