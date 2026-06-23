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

// Navatala C++ Wrapper Layer - Error Handling
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>
#include <exception>
#include <stdexcept>
#include <source_location>
#include <utility>

/// @file navatala/error.hpp
/// @brief Error handling mechanisms for the Navatala library.
///
/// This header provides:
/// - error_code enumeration for standardized error codes
/// - exception classes (when NAVATALA_ENABLE_EXCEPTIONS=1)
/// - expected<T> template for error-or-value returns
/// - expected<void> specialization for status returns
///
/// The error handling mode is configured at compile time via
/// NAVATALA_ENABLE_EXCEPTIONS. When exceptions are disabled,
/// error states are returned via expected<T> and accessing
/// values from error states will terminate the program.

namespace navatala {

// ============================================================================
// Error Codes
// ============================================================================

/// @brief Standardized error codes for the Navatala library.
///
/// These error codes are used consistently across the library to indicate
/// the type of error that occurred. They can be used with exceptions
/// (when enabled) or with expected<T> (when exceptions are disabled).
enum class error_code : uint32_t {
    success         = 0,   ///< No error, operation succeeded
    invalid_handle  = 1,   ///< Invalid or null handle passed to API
    invalid_param   = 2,   ///< Invalid parameter value
    gpu_error       = 3,   ///< Generic GPU-specific error (driver/runtime)
    out_of_memory   = 4,   ///< Failed to allocate memory (host or device)
    not_implemented = 5,   ///< Feature or operation not implemented
    io_error        = 6,   ///< I/O operation failed (file, network, etc.)
    runtime_error   = 7,   ///< General unexpected runtime issue
    overflow_error  = 8,   ///< Numeric overflow (e.g., index too large)
    timeout         = 9,   ///< Operation timed out
    cancelled       = 10,  ///< Operation was cancelled
    not_found       = 11,  ///< Requested resource not found
    already_exists  = 12,  ///< Resource already exists
    permission_denied = 13, ///< Permission denied for operation
    out_of_bounds   = 14,  ///< Index or offset out of bounds
    internal_error  = 99   ///< Unexpected library internal state
};

/// @brief Returns a string description of an error code.
[[nodiscard]] constexpr const char* error_code_name(error_code ec) noexcept {
    switch (ec) {
        case error_code::success:           return "success";
        case error_code::invalid_handle:    return "invalid_handle";
        case error_code::invalid_param:     return "invalid_param";
        case error_code::gpu_error:         return "gpu_error";
        case error_code::out_of_memory:     return "out_of_memory";
        case error_code::not_implemented:   return "not_implemented";
        case error_code::io_error:          return "io_error";
        case error_code::runtime_error:     return "runtime_error";
        case error_code::overflow_error:    return "overflow_error";
        case error_code::timeout:           return "timeout";
        case error_code::cancelled:         return "cancelled";
        case error_code::not_found:         return "not_found";
        case error_code::already_exists:    return "already_exists";
        case error_code::permission_denied: return "permission_denied";
        case error_code::out_of_bounds:     return "out_of_bounds";
        case error_code::internal_error:    return "internal_error";
        default:                            return "unknown_error";
    }
}

// ============================================================================
// Exception Classes (when NAVATALA_ENABLE_EXCEPTIONS=1)
// ============================================================================

#if NAVATALA_ENABLE_EXCEPTIONS

/// @brief Base exception class for Navatala errors.
///
/// This exception captures:
/// - An error_code for programmatic handling
/// - An operation description for context
/// - Source location (file, line, function) for debugging
///
/// Example:
/// @code
///   try {
///       auto res = resources::create(backend::cuda);
///   } catch (const navatala::exception& e) {
///       std::cerr << "Error: " << e.what() << "\n";
///       std::cerr << "Code: " << error_code_name(e.code()) << "\n";
///       std::cerr << "At: " << e.location().file_name() << ":"
///                 << e.location().line() << "\n";
///   }
/// @endcode
class exception : public std::runtime_error {
public:
    /// Constructs an exception with error details and source location.
    /// @param code The error code classifying this error.
    /// @param op A description of the operation that failed.
    /// @param loc The source location where the exception was thrown.
    exception(error_code code, std::string_view op,
              std::source_location loc = std::source_location::current())
        : std::runtime_error(format_message(code, op, loc))
        , code_(code)
        , operation_(op)
        , location_(loc)
    {}

    virtual ~exception() noexcept = default;

    /// Returns the error code associated with this exception.
    [[nodiscard]] error_code code() const noexcept { return code_; }

    /// Returns the operation description.
    [[nodiscard]] std::string_view operation() const noexcept { return operation_; }

    /// Returns the source location where the exception was thrown.
    [[nodiscard]] std::source_location location() const noexcept { return location_; }

private:
    static std::string format_message(error_code code, std::string_view op,
                                       const std::source_location& loc) {
        std::string msg;
        msg.reserve(256);
        msg += "[";
        msg += error_code_name(code);
        msg += "] ";
        msg += op;
        msg += " (at ";
        msg += loc.file_name();
        msg += ":";
        msg += std::to_string(loc.line());
        msg += " in ";
        msg += loc.function_name();
        msg += ")";
        return msg;
    }

    error_code code_;
    std::string operation_;
    std::source_location location_;
};

/// @brief Exception for invalid parameter errors.
class invalid_parameter : public exception {
public:
    /// Constructs an invalid_parameter exception.
    /// @param param_name The name of the invalid parameter.
    /// @param reason Why the parameter is invalid.
    /// @param loc Source location.
    invalid_parameter(std::string_view param_name, std::string_view reason,
                      std::source_location loc = std::source_location::current())
        : exception(error_code::invalid_param,
                    std::string(param_name) + ": " + std::string(reason), loc)
    {}
};

/// @brief Exception for GPU runtime errors.
class gpu_runtime_error : public exception {
public:
    /// Constructs a GPU runtime error exception.
    /// @param code The specific error code.
    /// @param op The operation that failed.
    /// @param details Additional error details (e.g., driver error string).
    /// @param loc Source location.
    gpu_runtime_error(error_code code, std::string_view op,
                      std::string_view details = "",
                      std::source_location loc = std::source_location::current())
        : exception(code,
                    details.empty() ? std::string(op)
                                    : std::string(op) + " - " + std::string(details),
                    loc)
    {}
};

/// @brief Exception for out-of-memory errors.
class out_of_memory_error : public exception {
public:
    /// Constructs an out-of-memory exception.
    /// @param op The allocation operation that failed.
    /// @param bytes The number of bytes requested.
    /// @param loc Source location.
    out_of_memory_error(std::string_view op, size_t bytes,
                        std::source_location loc = std::source_location::current())
        : exception(error_code::out_of_memory,
                    std::string(op) + " (requested " + std::to_string(bytes) + " bytes)",
                    loc)
    {}
};

#endif // NAVATALA_ENABLE_EXCEPTIONS

// ============================================================================
// Error Handling Macros
// ============================================================================

/// @brief Macro to throw an exception or abort depending on configuration.
///
/// When NAVATALA_ENABLE_EXCEPTIONS is 1, throws a navatala::exception.
/// When 0, prints the error and calls std::abort().
#if NAVATALA_ENABLE_EXCEPTIONS
#define NAVATALA_THROW_OR_ABORT(code, message) \
    throw ::navatala::exception((code), (message))
#else
#define NAVATALA_THROW_OR_ABORT(code, message) \
    do { \
        std::fprintf(stderr, "Navatala fatal error [%s]: %s\n", \
                     ::navatala::error_code_name(code), (message)); \
        std::abort(); \
    } while(0)
#endif

// ============================================================================
// Error Information Structure
// ============================================================================

/// @brief Structure holding detailed error information.
///
/// This is used internally by expected<T> to store error details
/// when the expected is in an error state.
struct error_info {
    error_code code;
    std::string operation;
    std::source_location location;

    error_info(error_code c, std::string_view op, std::source_location loc)
        : code(c), operation(op), location(loc) {}

    // Default constructor for empty/moved-from state
    error_info()
        : code(error_code::internal_error)
        , operation("uninitialized")
        , location(std::source_location::current()) {}
};

// ============================================================================
// expected<T> - Error-or-Value Type
// ============================================================================

/// @brief A type that represents either a value of type T or an error.
///
/// This class provides a mechanism for returning either a successful result
/// or an error without using exceptions. It uses a discriminated union
/// internally for space efficiency and correct lifetime management.
///
/// When exceptions are enabled (NAVATALA_ENABLE_EXCEPTIONS=1), accessing
/// the value when an error is present will throw an exception.
/// When exceptions are disabled, it will call std::terminate().
///
/// Example usage:
/// @code
///   expected<int> compute_value(bool succeed) {
///       if (succeed) {
///           return 42;  // Implicit conversion to expected<int>
///       }
///       return expected<int>(error_code::runtime_error, "computation failed");
///   }
///
///   auto result = compute_value(true);
///   if (result.has_value()) {
///       std::cout << "Value: " << result.value() << "\n";
///   } else {
///       std::cout << "Error: " << error_code_name(result.error()) << "\n";
///   }
/// @endcode
template<typename T>
class [[nodiscard]] expected {
    static_assert(!std::is_reference_v<T>, "expected<T&> is not supported");
    static_assert(!std::is_void_v<T>, "Use expected<void> specialization");

private:
    union storage_type {
        T value;
        error_info error;

        // Constructors/destructors managed externally
        storage_type() noexcept {}
        ~storage_type() noexcept {}
    };

    storage_type storage_;
    bool is_error_;

    // Helper to print fatal error and terminate
    [[noreturn]] void fatal_value_access() const {
        std::fputs("FATAL ERROR: Attempted to access value from an error-state expected object.\n"
                   "Error Code: ", stderr);
        std::fprintf(stderr, "%s (%u)\nOperation: %s\nFile: %s:%u\nFunction: %s\n",
                     error_code_name(storage_.error.code),
                     static_cast<unsigned>(storage_.error.code),
                     storage_.error.operation.c_str(),
                     storage_.error.location.file_name(),
                     storage_.error.location.line(),
                     storage_.error.location.function_name());
        std::terminate();
    }

    [[noreturn]] void fatal_error_access() const {
        std::fputs("FATAL ERROR: Attempted to access error from a value-state expected object.\n", stderr);
        std::terminate();
    }

public:
    using value_type = T;

    // -------------------- Constructors --------------------

    /// Constructs an expected holding a value.
    /// @param val The value to store.
    expected(T val) noexcept(std::is_nothrow_move_constructible_v<T>)
        : is_error_(false)
    {
        new (&storage_.value) T(std::move(val));
    }

    /// Constructs an expected holding an error.
    /// @param code The error code.
    /// @param op Description of the failed operation.
    /// @param loc Source location of the error.
    expected(error_code code, std::string_view op,
             std::source_location loc = std::source_location::current()) noexcept
        : is_error_(true)
    {
        new (&storage_.error) error_info(code, op, loc);
    }

    /// Constructs an expected from an error_info.
    expected(const error_info& err) noexcept
        : is_error_(true)
    {
        new (&storage_.error) error_info(err);
    }

    // -------------------- Destructor --------------------

    ~expected() noexcept {
        if (!is_error_) {
            storage_.value.~T();
        } else {
            storage_.error.~error_info();
        }
    }

    // -------------------- Copy Operations --------------------

    expected(const expected& other)
        : is_error_(other.is_error_)
    {
        if (!is_error_) {
            new (&storage_.value) T(other.storage_.value);
        } else {
            new (&storage_.error) error_info(other.storage_.error);
        }
    }

    expected& operator=(const expected& other) {
        if (this == &other) return *this;

        // Destroy current state
        this->~expected();

        // Copy new state
        is_error_ = other.is_error_;
        if (!is_error_) {
            new (&storage_.value) T(other.storage_.value);
        } else {
            new (&storage_.error) error_info(other.storage_.error);
        }
        return *this;
    }

    // -------------------- Move Operations --------------------

    expected(expected&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        : is_error_(other.is_error_)
    {
        if (!is_error_) {
            new (&storage_.value) T(std::move(other.storage_.value));
        } else {
            new (&storage_.error) error_info(std::move(other.storage_.error));
        }
    }

    expected& operator=(expected&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this == &other) return *this;

        // Destroy current state
        this->~expected();

        // Move new state
        is_error_ = other.is_error_;
        if (!is_error_) {
            new (&storage_.value) T(std::move(other.storage_.value));
        } else {
            new (&storage_.error) error_info(std::move(other.storage_.error));
        }
        return *this;
    }

    // -------------------- State Queries --------------------

    /// Returns true if this expected holds a value (success).
    [[nodiscard]] bool has_value() const noexcept { return !is_error_; }

    /// Returns true if this expected holds an error.
    [[nodiscard]] bool has_error() const noexcept { return is_error_; }

    /// Converts to bool: true if has value, false if has error.
    explicit operator bool() const noexcept { return !is_error_; }

    // -------------------- Value Access --------------------

    /// Returns a const reference to the contained value.
    /// @throws exception (if enabled) when in error state.
    /// @terminates (if exceptions disabled) when in error state.
    [[nodiscard]] const T& value() const& {
        if (is_error_) {
#if NAVATALA_ENABLE_EXCEPTIONS
            throw exception(storage_.error.code, storage_.error.operation, storage_.error.location);
#else
            fatal_value_access();
#endif
        }
        return storage_.value;
    }

    /// Returns a mutable reference to the contained value.
    [[nodiscard]] T& value() & {
        if (is_error_) {
#if NAVATALA_ENABLE_EXCEPTIONS
            throw exception(storage_.error.code, storage_.error.operation, storage_.error.location);
#else
            fatal_value_access();
#endif
        }
        return storage_.value;
    }

    /// Returns the contained value as an rvalue reference.
    [[nodiscard]] T&& value() && {
        if (is_error_) {
#if NAVATALA_ENABLE_EXCEPTIONS
            throw exception(storage_.error.code, storage_.error.operation, storage_.error.location);
#else
            fatal_value_access();
#endif
        }
        return std::move(storage_.value);
    }

    /// Returns the contained value, or a default if in error state.
    /// @param default_value The value to return if this holds an error.
    [[nodiscard]] T value_or(T default_value) const& noexcept(std::is_nothrow_copy_constructible_v<T>) {
        return is_error_ ? std::move(default_value) : storage_.value;
    }

    /// Returns the contained value, or a default if in error state (rvalue version).
    [[nodiscard]] T value_or(T default_value) && noexcept(std::is_nothrow_move_constructible_v<T>) {
        return is_error_ ? std::move(default_value) : std::move(storage_.value);
    }

    /// Pointer-like access to the value.
    [[nodiscard]] const T* operator->() const {
        if (is_error_) {
#if NAVATALA_ENABLE_EXCEPTIONS
            throw exception(storage_.error.code, storage_.error.operation, storage_.error.location);
#else
            fatal_value_access();
#endif
        }
        return &storage_.value;
    }

    [[nodiscard]] T* operator->() {
        if (is_error_) {
#if NAVATALA_ENABLE_EXCEPTIONS
            throw exception(storage_.error.code, storage_.error.operation, storage_.error.location);
#else
            fatal_value_access();
#endif
        }
        return &storage_.value;
    }

    /// Dereference to get the value.
    [[nodiscard]] const T& operator*() const& { return value(); }
    [[nodiscard]] T& operator*() & { return value(); }
    [[nodiscard]] T&& operator*() && { return std::move(*this).value(); }

    // -------------------- Error Access --------------------

    /// Returns the error code.
    /// Undefined behavior if no error is present.
    [[nodiscard]] error_code error() const noexcept {
        if (!is_error_) {
            fatal_error_access();
        }
        return storage_.error.code;
    }

    /// Returns the error info structure.
    /// Undefined behavior if no error is present.
    [[nodiscard]] const error_info& get_error_info() const noexcept {
        if (!is_error_) {
            fatal_error_access();
        }
        return storage_.error;
    }

    // -------------------- Monadic Operations --------------------

#if NAVATALA_ENABLE_EXCEPTIONS
    /// Returns the value, or throws an exception if in error state.
    /// Only available when exceptions are enabled.
    [[nodiscard]] T or_throw() && {
        if (is_error_) {
            throw exception(storage_.error.code, storage_.error.operation, storage_.error.location);
        }
        return std::move(storage_.value);
    }
#endif

    /// Applies a function to the value if present, propagating errors.
    /// @tparam F A callable taking T and returning expected<U>.
    /// @param f The function to apply.
    /// @return The result of f(value) or the propagated error.
    template<typename F>
    [[nodiscard]] auto and_then(F&& f) const& -> decltype(f(std::declval<const T&>())) {
        using result_type = decltype(f(std::declval<const T&>()));
        if (is_error_) {
            return result_type(storage_.error);
        }
        return f(storage_.value);
    }

    template<typename F>
    [[nodiscard]] auto and_then(F&& f) && -> decltype(f(std::declval<T&&>())) {
        using result_type = decltype(f(std::declval<T&&>()));
        if (is_error_) {
            return result_type(storage_.error);
        }
        return f(std::move(storage_.value));
    }

    /// Applies a function to transform the value if present.
    /// @tparam F A callable taking T and returning U.
    /// @param f The transformation function.
    /// @return expected<U> with the transformed value or the original error.
    template<typename F>
    [[nodiscard]] auto transform(F&& f) const& -> expected<decltype(f(std::declval<const T&>()))> {
        using U = decltype(f(std::declval<const T&>()));
        if (is_error_) {
            return expected<U>(storage_.error);
        }
        return expected<U>(f(storage_.value));
    }

    template<typename F>
    [[nodiscard]] auto transform(F&& f) && -> expected<decltype(f(std::declval<T&&>()))> {
        using U = decltype(f(std::declval<T&&>()));
        if (is_error_) {
            return expected<U>(storage_.error);
        }
        return expected<U>(f(std::move(storage_.value)));
    }
};

// ============================================================================
// expected<void> Specialization
// ============================================================================

/// @brief Specialization of expected for void, representing success or error.
///
/// This is useful for operations that don't return a value but may fail.
/// It acts as a lightweight status type.
///
/// Example:
/// @code
///   expected<void> do_operation() {
///       if (failure_condition) {
///           return expected<void>(error_code::runtime_error, "operation failed");
///       }
///       return expected<void>();  // Success
///   }
///
///   auto status = do_operation();
///   if (!status.has_value()) {
///       std::cerr << "Error: " << error_code_name(status.error()) << "\n";
///   }
/// @endcode
template<>
class [[nodiscard]] expected<void> {
private:
    error_info error_;
    bool is_error_;

    [[noreturn]] void fatal_value_access() const {
        std::fputs("FATAL ERROR: Attempted to check value from an error-state expected<void>.\n"
                   "Error Code: ", stderr);
        std::fprintf(stderr, "%s (%u)\nOperation: %s\nFile: %s:%u\n",
                     error_code_name(error_.code),
                     static_cast<unsigned>(error_.code),
                     error_.operation.c_str(),
                     error_.location.file_name(),
                     error_.location.line());
        std::terminate();
    }

    [[noreturn]] void fatal_error_access() const {
        std::fputs("FATAL ERROR: Attempted to access error from a success-state expected<void>.\n", stderr);
        std::terminate();
    }

public:
    using value_type = void;

    /// Constructs a successful status.
    expected() noexcept
        : error_(error_code::success, "success", std::source_location::current())
        , is_error_(false)
    {}

    /// Constructs an error status.
    /// @param code The error code.
    /// @param op Description of the failed operation.
    /// @param loc Source location.
    expected(error_code code, std::string_view op,
             std::source_location loc = std::source_location::current()) noexcept
        : error_(code, op, loc)
        , is_error_(code != error_code::success)
    {}

    /// Constructs from an error_info.
    expected(const error_info& err) noexcept
        : error_(err)
        , is_error_(err.code != error_code::success)
    {}

    /// Returns true if this represents success.
    [[nodiscard]] bool has_value() const noexcept { return !is_error_; }

    /// Returns true if this represents an error.
    [[nodiscard]] bool has_error() const noexcept { return is_error_; }

    /// Converts to bool: true if success, false if error.
    explicit operator bool() const noexcept { return !is_error_; }

    /// Checks success status. Throws or terminates if in error state.
    void value() const {
        if (!is_error_) return;
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_.code, error_.operation, error_.location);
#else
        fatal_value_access();
#endif
    }

    /// Returns the error code.
    /// Undefined behavior if no error is present.
    [[nodiscard]] error_code error() const noexcept {
        if (!is_error_) {
            fatal_error_access();
        }
        return error_.code;
    }

    /// Returns the error info structure.
    [[nodiscard]] const error_info& get_error_info() const noexcept {
        if (!is_error_) {
            fatal_error_access();
        }
        return error_;
    }

#if NAVATALA_ENABLE_EXCEPTIONS
    /// Throws if in error state (no-op if success).
    void or_throw() const {
        if (is_error_) {
            throw exception(error_.code, error_.operation, error_.location);
        }
    }
#endif

    /// Applies a function if in success state.
    template<typename F>
    [[nodiscard]] auto and_then(F&& f) const -> decltype(f()) {
        using result_type = decltype(f());
        if (is_error_) {
            return result_type(error_);
        }
        return f();
    }
};

// ============================================================================
// Type Aliases
// ============================================================================

/// @brief Alias for expected<T>, representing a result or error.
template<typename T>
using result = expected<T>;

/// @brief Alias for expected<void>, representing success or error status.
using status = expected<void>;

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Creates an error expected<T> from an error code.
/// @tparam T The expected value type.
/// @param code The error code.
/// @param op Description of the failed operation.
/// @param loc Source location.
template<typename T>
[[nodiscard]] expected<T> make_error(
    error_code code,
    std::string_view op,
    std::source_location loc = std::source_location::current()) noexcept
{
    return expected<T>(code, op, loc);
}

/// @brief Creates a success expected<void>.
[[nodiscard]] inline status make_success() noexcept {
    return status();
}

/// @brief Creates an error status.
/// @param code The error code.
/// @param op Description of the failed operation.
/// @param loc Source location.
[[nodiscard]] inline status make_error_status(
    error_code code,
    std::string_view op,
    std::source_location loc = std::source_location::current()) noexcept
{
    return status(code, op, loc);
}

// ============================================================================
// FFI Error Translation
// ============================================================================

/// @brief Translates an FFI error code to a C++ error_code.
///
/// This function provides a safe, centralized translation from FFI error codes
/// to the C++ error_code enum. It handles all known FFI error values and maps
/// unknown values to internal_error.
///
/// @param ffi_code The FFI error code from a C API call.
/// @return The corresponding C++ error_code.
[[nodiscard]] inline error_code translate_error(uint32_t ffi_code) noexcept {
    // FFI error codes match C++ error_code values by design, but we use
    // an explicit switch for safety and to handle any future divergence.
    switch (ffi_code) {
        case 0:  return error_code::success;
        case 1:  return error_code::invalid_handle;
        case 2:  return error_code::invalid_param;
        case 3:  return error_code::gpu_error;
        case 4:  return error_code::out_of_memory;
        case 5:  return error_code::not_implemented;
        case 6:  return error_code::io_error;
        case 7:  return error_code::runtime_error;
        case 8:  return error_code::overflow_error;
        case 9:  return error_code::timeout;
        case 10: return error_code::cancelled;
        case 11: return error_code::not_found;
        case 12: return error_code::already_exists;
        case 13: return error_code::permission_denied;
        case 14: return error_code::out_of_bounds;
        case 99: return error_code::internal_error;
        default: return error_code::internal_error;
    }
}

} // namespace navatala
