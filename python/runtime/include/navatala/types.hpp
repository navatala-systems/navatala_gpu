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

// Navatala C++ Wrapper Layer - Portable Type Definitions
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cstring>

/// @file navatala/types.hpp
/// @brief Portable fundamental types and utility functions for the Navatala library.
///
/// This header defines:
/// - float16_t: IEEE 754 half-precision floating point
/// - bfloat16_t: Brain floating point (Google's BF16)
/// - data_type enum: Runtime type discrimination
/// - Type traits and conversion utilities

namespace navatala {

// ============================================================================
// Forward Declarations for FFI
// ============================================================================

// FFI functions for float16/bfloat16 conversion (implemented in navatala_ffi.cpp)
extern "C" {
    uint16_t navatala_float_to_half(float f) noexcept;
    float navatala_half_to_float(uint16_t h) noexcept;
    uint16_t navatala_float_to_bfloat16(float f) noexcept;
    float navatala_bfloat16_to_float(uint16_t b) noexcept;
}

// ============================================================================
// Half-Precision Float (IEEE 754 binary16)
// ============================================================================

/// @brief Portable 16-bit float (IEEE 754 half precision).
///
/// This type provides a host-side representation of half-precision floats.
/// Arithmetic operations are emulated via float promotion on the host.
/// On GPU, native fp16 hardware is used where available.
///
/// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
/// Range: ~6.1e-5 to ~65504, with ~3 decimal digits of precision
struct float16_t {
    uint16_t data_;

    /// Default constructor (zero-initialized).
    constexpr float16_t() noexcept : data_(0) {}

    /// Construct from raw bits.
    explicit constexpr float16_t(uint16_t bits) noexcept : data_(bits) {}

    /// Convert from float32.
    /// @param f The float value to convert.
    /// @return The converted half-precision value.
    [[nodiscard]] static float16_t from_float(float f) noexcept {
        float16_t result;
        result.data_ = navatala_float_to_half(f);
        return result;
    }

    /// Convert to float32.
    /// @return The value as a float.
    [[nodiscard]] float to_float() const noexcept {
        return navatala_half_to_float(data_);
    }

    /// Get raw bits (e.g., for serialization).
    [[nodiscard]] constexpr uint16_t to_bits() const noexcept { return data_; }

    /// Construct from raw bits.
    [[nodiscard]] static constexpr float16_t from_bits(uint16_t bits) noexcept {
        return float16_t{bits};
    }

    // Arithmetic operators (via float promotion for host-side emulation)
    [[nodiscard]] float16_t operator+(float16_t other) const noexcept {
        return from_float(to_float() + other.to_float());
    }

    [[nodiscard]] float16_t operator-(float16_t other) const noexcept {
        return from_float(to_float() - other.to_float());
    }

    [[nodiscard]] float16_t operator*(float16_t other) const noexcept {
        return from_float(to_float() * other.to_float());
    }

    [[nodiscard]] float16_t operator/(float16_t other) const noexcept {
        return from_float(to_float() / other.to_float());
    }

    float16_t& operator+=(float16_t other) noexcept {
        *this = *this + other;
        return *this;
    }

    float16_t& operator-=(float16_t other) noexcept {
        *this = *this - other;
        return *this;
    }

    float16_t& operator*=(float16_t other) noexcept {
        *this = *this * other;
        return *this;
    }

    float16_t& operator/=(float16_t other) noexcept {
        *this = *this / other;
        return *this;
    }

    // Unary operators
    [[nodiscard]] float16_t operator-() const noexcept {
        // Flip sign bit
        return float16_t{static_cast<uint16_t>(data_ ^ 0x8000u)};
    }

    // Comparison operators
    [[nodiscard]] bool operator==(float16_t other) const noexcept {
        // Handle NaN: NaN != NaN
        // Handle +0 == -0
        if ((data_ & 0x7FFF) == 0 && (other.data_ & 0x7FFF) == 0) {
            return true; // Both are zero (positive or negative)
        }
        return data_ == other.data_;
    }

    [[nodiscard]] bool operator!=(float16_t other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] bool operator<(float16_t other) const noexcept {
        return to_float() < other.to_float();
    }

    [[nodiscard]] bool operator>(float16_t other) const noexcept {
        return to_float() > other.to_float();
    }

    [[nodiscard]] bool operator<=(float16_t other) const noexcept {
        return to_float() <= other.to_float();
    }

    [[nodiscard]] bool operator>=(float16_t other) const noexcept {
        return to_float() >= other.to_float();
    }

    // Conversion operator to float (explicit to avoid accidental conversions)
    explicit operator float() const noexcept {
        return to_float();
    }
};

// ============================================================================
// Brain Float 16 (bfloat16)
// ============================================================================

/// @brief Portable bfloat16 (Brain Float16).
///
/// bfloat16 is a 16-bit floating point format developed by Google Brain.
/// It has the same exponent range as float32 but reduced precision.
/// This makes it well-suited for deep learning where dynamic range is
/// more important than precision.
///
/// Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
/// Range: Same as float32 (~1.2e-38 to ~3.4e38)
/// Precision: ~2-3 decimal digits
struct bfloat16_t {
    uint16_t data_;

    /// Default constructor (zero-initialized).
    constexpr bfloat16_t() noexcept : data_(0) {}

    /// Construct from raw bits.
    explicit constexpr bfloat16_t(uint16_t bits) noexcept : data_(bits) {}

    /// Convert from float32.
    /// @param f The float value to convert.
    /// @return The converted bfloat16 value.
    [[nodiscard]] static bfloat16_t from_float(float f) noexcept {
        bfloat16_t result;
        result.data_ = navatala_float_to_bfloat16(f);
        return result;
    }

    /// Convert to float32.
    /// @return The value as a float.
    [[nodiscard]] float to_float() const noexcept {
        return navatala_bfloat16_to_float(data_);
    }

    /// Get raw bits (e.g., for serialization).
    [[nodiscard]] constexpr uint16_t to_bits() const noexcept { return data_; }

    /// Construct from raw bits.
    [[nodiscard]] static constexpr bfloat16_t from_bits(uint16_t bits) noexcept {
        return bfloat16_t{bits};
    }

    // Arithmetic operators (via float promotion for host-side emulation)
    [[nodiscard]] bfloat16_t operator+(bfloat16_t other) const noexcept {
        return from_float(to_float() + other.to_float());
    }

    [[nodiscard]] bfloat16_t operator-(bfloat16_t other) const noexcept {
        return from_float(to_float() - other.to_float());
    }

    [[nodiscard]] bfloat16_t operator*(bfloat16_t other) const noexcept {
        return from_float(to_float() * other.to_float());
    }

    [[nodiscard]] bfloat16_t operator/(bfloat16_t other) const noexcept {
        return from_float(to_float() / other.to_float());
    }

    bfloat16_t& operator+=(bfloat16_t other) noexcept {
        *this = *this + other;
        return *this;
    }

    bfloat16_t& operator-=(bfloat16_t other) noexcept {
        *this = *this - other;
        return *this;
    }

    bfloat16_t& operator*=(bfloat16_t other) noexcept {
        *this = *this * other;
        return *this;
    }

    bfloat16_t& operator/=(bfloat16_t other) noexcept {
        *this = *this / other;
        return *this;
    }

    // Unary operators
    [[nodiscard]] bfloat16_t operator-() const noexcept {
        // Flip sign bit
        return bfloat16_t{static_cast<uint16_t>(data_ ^ 0x8000u)};
    }

    // Comparison operators
    [[nodiscard]] bool operator==(bfloat16_t other) const noexcept {
        // Handle +0 == -0
        if ((data_ & 0x7FFF) == 0 && (other.data_ & 0x7FFF) == 0) {
            return true;
        }
        return data_ == other.data_;
    }

    [[nodiscard]] bool operator!=(bfloat16_t other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] bool operator<(bfloat16_t other) const noexcept {
        return to_float() < other.to_float();
    }

    [[nodiscard]] bool operator>(bfloat16_t other) const noexcept {
        return to_float() > other.to_float();
    }

    [[nodiscard]] bool operator<=(bfloat16_t other) const noexcept {
        return to_float() <= other.to_float();
    }

    [[nodiscard]] bool operator>=(bfloat16_t other) const noexcept {
        return to_float() >= other.to_float();
    }

    // Conversion operator to float (explicit to avoid accidental conversions)
    explicit operator float() const noexcept {
        return to_float();
    }
};

// ============================================================================
// Type Traits
// ============================================================================

/// Type trait to detect float16_t.
template<typename T>
struct is_fp16 : std::false_type {};

template<>
struct is_fp16<float16_t> : std::true_type {};

/// Helper variable template for is_fp16.
template<typename T>
inline constexpr bool is_fp16_v = is_fp16<T>::value;

/// Type trait to detect bfloat16_t.
template<typename T>
struct is_bf16 : std::false_type {};

template<>
struct is_bf16<bfloat16_t> : std::true_type {};

/// Helper variable template for is_bf16.
template<typename T>
inline constexpr bool is_bf16_v = is_bf16<T>::value;

/// Helper for static_assert in if constexpr branches.
/// Usage: static_assert(always_false_v<T>, "message");
template<typename T>
inline constexpr bool always_false_v = false;

// ============================================================================
// Data Type Enumeration
// ============================================================================

/// @brief Enumeration of supported data types on the GPU.
///
/// This enum provides runtime type discrimination for buffers, datasets,
/// and other GPU data structures. Values are stable and match FFI conventions.
enum class data_type : uint32_t {
    float32 = 0,   ///< 32-bit IEEE 754 floating point
    float16 = 1,   ///< 16-bit IEEE 754 floating point (half)
    bf16    = 2,   ///< 16-bit Brain floating point
    int8    = 3,   ///< 8-bit signed integer
    uint8   = 4,   ///< 8-bit unsigned integer
    int32   = 5,   ///< 32-bit signed integer
    uint32  = 6,   ///< 32-bit unsigned integer
    int64   = 7,   ///< 64-bit signed integer
    uint64  = 8,   ///< 64-bit unsigned integer
    float64 = 9,   ///< 64-bit IEEE 754 floating point (double)
    int16   = 10,  ///< 16-bit signed integer
    uint16  = 11,  ///< 16-bit unsigned integer
};

/// @brief Returns the size in bytes of a data_type.
[[nodiscard]] constexpr size_t sizeof_data_type(data_type dt) noexcept {
    switch (dt) {
        case data_type::float32: return 4;
        case data_type::float16: return 2;
        case data_type::bf16:    return 2;
        case data_type::int8:    return 1;
        case data_type::uint8:   return 1;
        case data_type::int32:   return 4;
        case data_type::uint32:  return 4;
        case data_type::int64:   return 8;
        case data_type::uint64:  return 8;
        case data_type::float64: return 8;
        case data_type::int16:   return 2;
        case data_type::uint16:  return 2;
        default:                 return 0;
    }
}

/// @brief Returns a string name for a data_type.
[[nodiscard]] constexpr const char* data_type_name(data_type dt) noexcept {
    switch (dt) {
        case data_type::float32: return "float32";
        case data_type::float16: return "float16";
        case data_type::bf16:    return "bf16";
        case data_type::int8:    return "int8";
        case data_type::uint8:   return "uint8";
        case data_type::int32:   return "int32";
        case data_type::uint32:  return "uint32";
        case data_type::int64:   return "int64";
        case data_type::uint64:  return "uint64";
        case data_type::float64: return "float64";
        case data_type::int16:   return "int16";
        case data_type::uint16:  return "uint16";
        default:                 return "unknown";
    }
}

/// @brief Converts a C++ type to its corresponding data_type enum value at compile time.
///
/// @tparam T The C++ type to convert.
/// @return The corresponding data_type value.
///
/// Example:
/// @code
///   auto dt = to_data_type<float>();  // returns data_type::float32
///   auto dt2 = to_data_type<float16_t>();  // returns data_type::float16
/// @endcode
template<typename T>
[[nodiscard]] constexpr data_type to_data_type() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return data_type::float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return data_type::float64;
    } else if constexpr (std::is_same_v<T, float16_t>) {
        return data_type::float16;
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return data_type::bf16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return data_type::int8;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return data_type::uint8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return data_type::int16;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return data_type::uint16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return data_type::int32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return data_type::uint32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return data_type::int64;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return data_type::uint64;
    } else {
        static_assert(always_false_v<T>, "Unsupported data type for to_data_type()");
    }
}

/// Helper variable template for to_data_type.
template<typename T>
inline constexpr data_type to_data_type_v = to_data_type<T>();

// ============================================================================
// Distance Metric Types
// ============================================================================

/// @brief Distance metric types for vector search operations.
///
/// These metrics define how similarity/distance is computed between vectors
/// in nearest-neighbor search operations.
enum class distance_type : uint32_t {
    l2_expanded      = 0,  ///< Squared L2 (Euclidean) distance, expanded form
    l2_sqrt_expanded = 1,  ///< L2 distance (sqrt of squared), expanded form
    cosine_expanded  = 2,  ///< Cosine similarity (converted to distance), expanded
    l1               = 3,  ///< L1 (Manhattan) distance
    l2_unexpanded    = 4,  ///< Squared L2 distance, unexpanded form
    l2_sqrt_unexpanded = 5, ///< L2 distance, unexpanded form
    inner_product    = 6,  ///< Inner product (negative for similarity)
    linf             = 7,  ///< L-infinity (Chebyshev) distance
    canberra         = 8,  ///< Canberra distance
    lp               = 9,  ///< General Lp distance (requires p parameter)
    hamming          = 10, ///< Hamming distance (for binary vectors)
    jensen_shannon   = 11, ///< Jensen-Shannon divergence
    hellinger        = 12, ///< Hellinger distance
    correlation      = 13, ///< Correlation distance
};

/// @brief Returns a string name for a distance_type.
[[nodiscard]] constexpr const char* distance_type_name(distance_type dt) noexcept {
    switch (dt) {
        case distance_type::l2_expanded:      return "l2_expanded";
        case distance_type::l2_sqrt_expanded: return "l2_sqrt_expanded";
        case distance_type::cosine_expanded:  return "cosine_expanded";
        case distance_type::l1:               return "l1";
        case distance_type::l2_unexpanded:    return "l2_unexpanded";
        case distance_type::l2_sqrt_unexpanded: return "l2_sqrt_unexpanded";
        case distance_type::inner_product:    return "inner_product";
        case distance_type::linf:             return "linf";
        case distance_type::canberra:         return "canberra";
        case distance_type::lp:               return "lp";
        case distance_type::hamming:          return "hamming";
        case distance_type::jensen_shannon:   return "jensen_shannon";
        case distance_type::hellinger:        return "hellinger";
        case distance_type::correlation:      return "correlation";
        default:                              return "unknown";
    }
}

// ============================================================================
// Index Type
// ============================================================================

/// @brief Index type for search results.
///
/// This type is used for array indices and search result indices.
/// Using uint64_t allows for very large datasets (up to ~18 quintillion elements).
using index_type = uint64_t;

} // namespace navatala
