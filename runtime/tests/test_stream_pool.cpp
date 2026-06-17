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

// GPU Runtime - Stream Pool Tests
// Tests for stream_pool.h functionality

#include "../include/gpu_runtime.h"
#include "../include/stream_pool.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>

using namespace GpuRuntime;

// ============================================================================
// Basic Stream Pool Tests
// ============================================================================

void test_stream_pool_create() {
    std::cout << "=== Test: Stream Pool Creation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool(*device, 4);

    assert(pool.available() == 4 && "Pool should have 4 available queues");
    assert(pool.total() == 4 && "Pool should have 4 total queues");

    std::cout << "PASS: Stream pool creation" << std::endl;
}

void test_stream_pool_acquire_release() {
    std::cout << "=== Test: Stream Pool Acquire/Release ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool(*device, 2);
    assert(pool.available() == 2);

    // Acquire a queue
    auto q1 = pool.acquire();
    assert(q1 != nullptr && "Acquire should return a valid queue");
    assert(pool.available() == 1 && "Available should decrease after acquire");

    // Acquire another
    auto q2 = pool.acquire();
    assert(q2 != nullptr);
    assert(pool.available() == 0 && "Pool should be empty");

    // Acquire when empty (should create a new one)
    auto q3 = pool.acquire();
    assert(q3 != nullptr && "Should be able to acquire even when pool is empty");
    assert(pool.total() == 3 && "Total should increase when pool needs to grow");

    // Release queues back
    pool.release(std::move(q1));
    assert(pool.available() == 1);

    pool.release(std::move(q2));
    assert(pool.available() == 2);

    pool.release(std::move(q3));
    assert(pool.available() == 3);

    std::cout << "PASS: Stream pool acquire/release" << std::endl;
}

void test_stream_pool_priority() {
    std::cout << "=== Test: Stream Pool Priority ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Create pool with High priority default
    StreamPool pool(*device, 2, StreamPriority::High);

    // Acquire with default priority (High)
    auto q1 = pool.acquire();
    assert(q1 != nullptr);

    // Acquire with specific priority
    auto q2 = pool.acquire(StreamPriority::Low);
    assert(q2 != nullptr);

    pool.release(std::move(q1));
    pool.release(std::move(q2));

    std::cout << "PASS: Stream pool priority" << std::endl;
}

void test_stream_pool_use_queue() {
    std::cout << "=== Test: Stream Pool Use Queue ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool(*device, 2);

    auto queue = pool.acquire();
    assert(queue != nullptr);

    // Use the queue for actual operations
    auto buffer1 = device->createBuffer(1024, MemoryKind::Device);
    auto buffer2 = device->createBuffer(1024, MemoryKind::Device);
    assert(buffer1 != nullptr && buffer2 != nullptr);

    // Memcpy should work with pooled queue
    queue->memcpy(*buffer2, *buffer1, 1024);
    queue->synchronize();

    pool.release(std::move(queue));

    std::cout << "PASS: Stream pool use queue" << std::endl;
}

void test_stream_pool_trim() {
    std::cout << "=== Test: Stream Pool Trim ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool(*device, 4);
    assert(pool.available() == 4);
    size_t original_total = pool.total();

    // Trim to 2 - available decreases, but total() tracks "ever created"
    pool.trim(2);
    assert(pool.available() == 2 && "Should have 2 available after trim");
    // total() tracks total ever created, not current count
    assert(pool.total() == original_total && "Total should remain unchanged (tracks created, not current)");

    // Trim to 0
    pool.trim(0);
    assert(pool.available() == 0);

    std::cout << "PASS: Stream pool trim" << std::endl;
}

void test_stream_pool_clear() {
    std::cout << "=== Test: Stream Pool Clear ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool(*device, 4);

    // Acquire some queues
    auto q1 = pool.acquire();
    auto q2 = pool.acquire();
    assert(pool.available() == 2);

    // Clear available pool (should not affect acquired queues)
    pool.clear();
    assert(pool.available() == 0);

    // The acquired queues should still work
    assert(q1 != nullptr);
    q1->synchronize();

    // Release should work (adds back to empty pool)
    pool.release(std::move(q1));
    pool.release(std::move(q2));
    assert(pool.available() == 2);

    std::cout << "PASS: Stream pool clear" << std::endl;
}

void test_stream_pool_move() {
    std::cout << "=== Test: Stream Pool Move ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool1(*device, 4);
    assert(pool1.available() == 4);

    // Move construct
    StreamPool pool2(std::move(pool1));
    assert(pool2.available() == 4);

    // pool1 should be empty/invalid after move
    // (exact behavior depends on implementation)

    // Move assign
    StreamPool pool3(*device, 2);
    pool3 = std::move(pool2);
    assert(pool3.available() == 4);

    std::cout << "PASS: Stream pool move" << std::endl;
}

void test_stream_pool_multiple_devices() {
    std::cout << "=== Test: Stream Pool Multiple Devices ===" << std::endl;

    auto device0 = Device::create(0);
    if (!device0) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Try to create second device (may not exist)
    auto device1 = Device::create(1);

    StreamPool pool0(*device0, 2);
    assert(pool0.available() == 2);

    if (device1) {
        StreamPool pool1(*device1, 2);
        assert(pool1.available() == 2);

        // Both pools should work independently
        auto q0 = pool0.acquire();
        auto q1 = pool1.acquire();
        assert(q0 != nullptr && q1 != nullptr);

        pool0.release(std::move(q0));
        pool1.release(std::move(q1));

        std::cout << "PASS: Stream pool multiple devices (2 devices)" << std::endl;
    } else {
        std::cout << "PASS: Stream pool multiple devices (1 device only)" << std::endl;
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

void test_stream_pool_stress() {
    std::cout << "=== Test: Stream Pool Stress ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    StreamPool pool(*device, 2);

    // Rapidly acquire and release many times
    for (int i = 0; i < 100; ++i) {
        auto q = pool.acquire();
        assert(q != nullptr);
        pool.release(std::move(q));
    }

    // Acquire multiple, then release all
    std::vector<std::unique_ptr<Queue>> queues;
    for (int i = 0; i < 10; ++i) {
        queues.push_back(pool.acquire());
        assert(queues.back() != nullptr);
    }

    for (auto& q : queues) {
        pool.release(std::move(q));
    }

    std::cout << "PASS: Stream pool stress" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Runtime Stream Pool Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    test_stream_pool_create();
    test_stream_pool_acquire_release();
    test_stream_pool_priority();
    test_stream_pool_use_queue();
    test_stream_pool_trim();
    test_stream_pool_clear();
    test_stream_pool_move();
    test_stream_pool_multiple_devices();
    test_stream_pool_stress();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Stream Pool Tests Completed!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
