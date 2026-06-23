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

// Navatala C++ Wrapper Layer - Main Include Header
// Part of the Navatala GPU Runtime Library
#pragma once

/// @file navatala/navatala.hpp
/// @brief Main include header for the Navatala C++ wrapper library.
///
/// This header provides a convenient single-include entry point for the
/// Navatala GPU compute library. Including this header gives access to
/// all core functionality.
///
/// # Library Overview
///
/// Navatala is a modern C++20 wrapper layer that provides:
/// - Type-safe, RAII-managed GPU resources
/// - Zero-cost abstractions for GPU compute operations
/// - Backend-transparent API supporting CUDA, HIP, Metal, Vulkan, and OpenCL
/// - Vector search operations (CAGRA, IVF-PQ, etc.)
/// - Neural operator support for CFD/ML workloads (experimental)
///
/// # Quick Start
///
/// @code
/// #include <navatala/navatala.hpp>
///
/// int main() {
///     // Create GPU resources (auto-selects best backend)
///     auto res = navatala::resources::create();
///     if (!res.has_value()) {
///         std::cerr << "Failed to create GPU context\n";
///         return 1;
///     }
///
///     // Use the resources
///     auto& gpu = res.value();
///     std::cout << "Using backend: " << navatala::backend_name(gpu.get_backend()) << "\n";
///
///     return 0;
/// }
/// @endcode
///
/// # Error Handling
///
/// The library supports two error handling modes, controlled by the
/// NAVATALA_ENABLE_EXCEPTIONS macro:
///
/// - **Exceptions enabled (default):** Functions may throw `navatala::exception`
///   or derived types. Use try-catch for error handling.
///
/// - **Exceptions disabled:** Functions return `navatala::expected<T>` which
///   contains either a value or an error. Check with `has_value()` before
///   accessing the value.
///
/// # Configuration
///
/// Compile-time configuration is available via macros defined before including
/// any navatala headers:
///
/// - `NAVATALA_ENABLE_EXCEPTIONS` - Enable/disable exception throwing (default: 1)
/// - `NAVATALA_DEFAULT_BACKEND` - Default GPU backend to use
/// - `NAVATALA_DEFAULT_MEMORY_POOL` - Enable default memory pooling (default: 1)
/// - `NAVATALA_LOG_LEVEL` - Logging verbosity (0=off, 4=debug)
///
/// See navatala/config.hpp for full configuration options.
///
/// # Thread Safety
///
/// - `resources` instances are not thread-safe; use one per thread or synchronize.
/// - `queue` operations from the same queue must be serialized.
/// - Different queues can be used concurrently from different threads.
/// - Memory pools are thread-safe by default.
/// - Read operations on immutable data are always thread-safe.
///
/// # Version Information
///
/// @see NAVATALA_VERSION_MAJOR, NAVATALA_VERSION_MINOR, NAVATALA_VERSION_PATCH

// ============================================================================
// Core Headers ()
// ============================================================================

#include "config.hpp"   // Compile-time configuration
#include "types.hpp"    // Portable types (float16_t, bfloat16_t, data_type)
#include "error.hpp"    // Error handling (error_code, exception, expected<T>)
#include "backend.hpp"  // Backend enumeration and capabilities

// ============================================================================
// Resource Management Headers ()
// ============================================================================

#include "queue.hpp"     // Queue and event abstractions
#include "memory.hpp"    // Memory resource and pools
#include "resource.hpp"  // GPU context management (depends on queue, memory)

// ============================================================================
// Data Management Headers ()
// ============================================================================

#include "buffer.hpp"    // Device buffer
#include "dataset.hpp"   // Structured datasets
#include "linalg.hpp"    // Portable dense/sparse linear algebra facades

// ============================================================================
// Vector Search Headers ()
// ============================================================================

#include "results.hpp"   // Search results
#include "index.hpp"     // Index types (CAGRA, IVF-PQ, etc.)
#include "any_index.hpp" // Type-erased index
#include "batch.hpp"     // Batch operations
#include "kmeans.hpp"    // K-means clustering
#include "distance.hpp"  // Distance computations

// ============================================================================
// Neural Operators Headers (- EXPERIMENTAL)
// ============================================================================

#include "neural_operators/common.hpp"
#include "neural_operators/fno.hpp"
#include "neural_operators/pino.hpp"
#include "neural_operators/mesh_graph_net.hpp"

// ============================================================================
// Profiling and Metrics Headers ()
// ============================================================================

#include "profiling.hpp"  // Performance instrumentation
#include "metrics.hpp"    // Runtime statistics

// ============================================================================
// Namespace Documentation
// ============================================================================

/// @namespace navatala
/// @brief Root namespace for the Navatala GPU compute library.
///
/// The navatala namespace contains all public API types and functions for
/// GPU-accelerated computing. Key components include:
///
/// - **Resource Management:** `resources`, `queue`, `event`, `memory::pool`
/// - **Data Types:** `float16_t`, `bfloat16_t`, `data_type`, `distance_type`
/// - **Error Handling:** `error_code`, `exception`, `expected<T>`, `status`
/// - **Buffers:** `buffer`, `dataset`
/// - **Vector Search:** Index types, search operations, batch processing
///
/// @see navatala::neural_operators for experimental neural operator support.

/// @namespace navatala::memory
/// @brief Memory management utilities.
///
/// Contains memory resource abstractions and allocation pools.

/// @namespace navatala::neural_operators
/// @brief Experimental neural operator wrappers (PhysicsNeMo integration).
///
/// @warning This namespace contains experimental APIs that may change
/// without notice between versions. Use with caution in production code.
///
/// Neural operators provide GPU-accelerated implementations of:
/// - Fourier Neural Operator (FNO)
/// - Physics-Informed Neural Operator (PINO)
/// - MeshGraphNet for mesh-based simulations
