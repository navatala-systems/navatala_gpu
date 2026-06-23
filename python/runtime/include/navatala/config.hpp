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

// Navatala C++ Wrapper Layer - Compile-Time Configuration
// Part of the Navatala GPU Runtime Library
#pragma once

/// @file navatala/config.hpp
/// @brief Compile-time configuration macros for the Navatala C++ wrapper layer.
///
/// This header defines macros for customizing library features at compile time.
/// Users can override these defaults by defining the macros before including
/// any navatala headers, or via compiler flags (e.g., -DNAVATALA_ENABLE_EXCEPTIONS=0).

// ============================================================================
// Exception Handling Mode
// ============================================================================

/// @def NAVATALA_ENABLE_EXCEPTIONS
/// @brief Controls whether the library uses C++ exceptions for error handling.
///
/// - NAVATALA_ENABLE_EXCEPTIONS=1 (default): Use exceptions for error reporting.
///   Functions may throw navatala::exception or derived types.
/// - NAVATALA_ENABLE_EXCEPTIONS=0: Use error codes / expected<T, E>.
///   No exceptions are thrown; errors are returned via expected<T>.
///
/// When exceptions are disabled, accessing a value from an error-state
/// expected<T> will call std::terminate() with a diagnostic message.
#ifndef NAVATALA_ENABLE_EXCEPTIONS
#define NAVATALA_ENABLE_EXCEPTIONS 1
#endif

// ============================================================================
// RTTI Support
// ============================================================================

/// @def NAVATALA_ENABLE_RTTI
/// @brief Controls whether RTTI (Run-Time Type Information) is required.
///
/// - NAVATALA_ENABLE_RTTI=0 (default): RTTI is not required.
///   The library uses std::variant visitation which does not need RTTI.
/// - NAVATALA_ENABLE_RTTI=1: Enable if external dependencies require RTTI.
///
/// Note: Disabling RTTI can reduce binary size and improve performance
/// in some scenarios.
#ifndef NAVATALA_ENABLE_RTTI
#define NAVATALA_ENABLE_RTTI 0
#endif

// ============================================================================
// Backend Selection
// ============================================================================

/// Backend numeric identifiers for compile-time and runtime selection.
/// These values are stable and should not be changed.
#define NAVATALA_BACKEND_CUDA   0
#define NAVATALA_BACKEND_HIP    1
#define NAVATALA_BACKEND_METAL  2
#define NAVATALA_BACKEND_OPENCL 3
#define NAVATALA_BACKEND_VULKAN 4
#define NAVATALA_BACKEND_AUTO   255

/// @def NAVATALA_DEFAULT_BACKEND
/// @brief Specifies the default GPU backend to use when not explicitly specified.
///
/// Valid values:
/// - NAVATALA_BACKEND_AUTO (255): Auto-select the best available backend.
/// - NAVATALA_BACKEND_CUDA (0): Use NVIDIA CUDA.
/// - NAVATALA_BACKEND_HIP (1): Use AMD HIP/ROCm.
/// - NAVATALA_BACKEND_METAL (2): Use Apple Metal.
/// - NAVATALA_BACKEND_OPENCL (3): Use OpenCL.
/// - NAVATALA_BACKEND_VULKAN (4): Use Vulkan Compute.
///
/// Default: NAVATALA_BACKEND_AUTO
#ifndef NAVATALA_DEFAULT_BACKEND
#define NAVATALA_DEFAULT_BACKEND NAVATALA_BACKEND_AUTO
#endif

// ============================================================================
// Memory Pool Configuration
// ============================================================================

/// @def NAVATALA_DEFAULT_MEMORY_POOL
/// @brief Controls whether a default memory pool is used for allocations.
///
/// - NAVATALA_DEFAULT_MEMORY_POOL=1 (default): Use a default memory pool.
///   This typically improves allocation performance by reducing the number
///   of underlying GPU memory allocation calls.
/// - NAVATALA_DEFAULT_MEMORY_POOL=0: Use direct GPU allocations.
///   Each allocation goes directly to the GPU driver/runtime.
#ifndef NAVATALA_DEFAULT_MEMORY_POOL
#define NAVATALA_DEFAULT_MEMORY_POOL 1
#endif

/// @def NAVATALA_DEFAULT_POOL_INITIAL_SIZE
/// @brief Initial size of the default memory pool in bytes.
/// Default: 256 MB
#ifndef NAVATALA_DEFAULT_POOL_INITIAL_SIZE
#define NAVATALA_DEFAULT_POOL_INITIAL_SIZE (256ULL * 1024ULL * 1024ULL)
#endif

/// @def NAVATALA_DEFAULT_POOL_MAX_SIZE
/// @brief Maximum size the default memory pool can grow to in bytes.
/// Default: 0 (unlimited, grows as needed up to device memory)
#ifndef NAVATALA_DEFAULT_POOL_MAX_SIZE
#define NAVATALA_DEFAULT_POOL_MAX_SIZE 0
#endif

// ============================================================================
// Logging Configuration
// ============================================================================

/// @def NAVATALA_LOG_LEVEL
/// @brief Compile-time logging level for internal diagnostics.
///
/// Levels:
/// - 0: Off (no logging)
/// - 1: Error only
/// - 2: Warning and above (default)
/// - 3: Info and above
/// - 4: Debug (verbose)
///
/// Note: Higher log levels may impact performance.
#ifndef NAVATALA_LOG_LEVEL
#define NAVATALA_LOG_LEVEL 2
#endif

// ============================================================================
// Debug and Assertions
// ============================================================================

/// @def NAVATALA_DEBUG
/// @brief Enable debug assertions and extra validation.
///
/// When enabled, additional runtime checks are performed to catch
/// programming errors early. This may impact performance.
#ifndef NAVATALA_DEBUG
#ifdef NDEBUG
#define NAVATALA_DEBUG 0
#else
#define NAVATALA_DEBUG 1
#endif
#endif

// ============================================================================
// API Export/Import Macros
// ============================================================================

/// @def NAVATALA_API
/// @brief Macro for API symbol visibility (shared library export/import).
#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef NAVATALA_BUILDING_SHARED
        #define NAVATALA_API __declspec(dllexport)
    #elif defined(NAVATALA_USING_SHARED)
        #define NAVATALA_API __declspec(dllimport)
    #else
        #define NAVATALA_API
    #endif
#else
    #if __GNUC__ >= 4 || defined(__clang__)
        #define NAVATALA_API __attribute__((visibility("default")))
    #else
        #define NAVATALA_API
    #endif
#endif

// ============================================================================
// Version Information
// ============================================================================

/// @def NAVATALA_VERSION_MAJOR
/// @def NAVATALA_VERSION_MINOR
/// @def NAVATALA_VERSION_PATCH
/// @brief Library version numbers following semantic versioning.
#define NAVATALA_VERSION_MAJOR 0
#define NAVATALA_VERSION_MINOR 1
#define NAVATALA_VERSION_PATCH 0

/// @def NAVATALA_VERSION_STRING
/// @brief Library version as a string literal.
#define NAVATALA_VERSION_STRING "0.1.0"

// ============================================================================
// Feature Detection
// ============================================================================

// C++20 feature checks
#if __cplusplus < 202002L
    #error "Navatala requires C++20 or later. Please compile with -std=c++20 or equivalent."
#endif

// Check for std::source_location support
#if __has_include(<source_location>)
    #define NAVATALA_HAS_SOURCE_LOCATION 1
#else
    #define NAVATALA_HAS_SOURCE_LOCATION 0
    #error "Navatala requires <source_location> (C++20). Please use a compatible compiler."
#endif

// Check for std::span support
#if __has_include(<span>)
    #define NAVATALA_HAS_SPAN 1
#else
    #define NAVATALA_HAS_SPAN 0
    #error "Navatala requires <span> (C++20). Please use a compatible compiler."
#endif

// Check for concepts support
#ifdef __cpp_concepts
    #if __cpp_concepts >= 202002L
        #define NAVATALA_HAS_CONCEPTS 1
    #else
        #define NAVATALA_HAS_CONCEPTS 0
    #endif
#else
    #define NAVATALA_HAS_CONCEPTS 0
#endif
