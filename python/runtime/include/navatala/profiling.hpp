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

// Navatala C++ Wrapper Layer - Profiling and Performance Instrumentation
// Part of the Navatala GPU Runtime Library
// Polish

#pragma once

#include "config.hpp"
#include "error.hpp"
#include "queue.hpp"
#include "resource.hpp"
#include "navatala_ffi.h"

#include <chrono>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <functional>
#include <optional>
#include <atomic>
#include <mutex>

/// @file navatala/profiling.hpp
/// @brief GPU profiling and performance instrumentation utilities.
///
/// This header provides:
/// - scoped_timer: RAII-based CPU/GPU timing measurement
/// - event_marker: GPU timeline markers for debugging/tracing
/// - profiler: Central profiling coordinator
/// - Timing utilities for benchmarking GPU operations
///
/// @section profiling_usage Usage
///
/// Basic profiling:
/// @code
///   auto res = navatala::resources::create().value();
///   navatala::profiling::profiler prof(res);
///
///   {
///       auto timer = prof.scope("matrix_multiply");
///       // ... GPU operations ...
///   } // Timing recorded automatically
///
///   prof.report();
/// @endcode
///
/// GPU event timing:
/// @code
///   auto start_event = navatala::event::create(res).value();
///   auto end_event = navatala::event::create(res).value();
///
///   queue.record(start_event);
///   // ... GPU operations ...
///   queue.record(end_event);
///   queue.sync();
///
///   float ms = navatala::profiling::elapsed_ms(start_event, end_event);
/// @endcode

namespace navatala {
namespace profiling {

// ============================================================================
// Timing Record
// ============================================================================

/// @brief A single timing measurement record.
struct timing_record {
    std::string name;                                       ///< Operation name
    std::chrono::steady_clock::time_point start_time;       ///< CPU start time
    std::chrono::steady_clock::time_point end_time;         ///< CPU end time
    std::optional<float> gpu_elapsed_ms;                    ///< GPU time (if measured)
    uint64_t bytes_processed{0};                            ///< Bytes processed (optional)
    uint32_t iteration{0};                                  ///< Iteration number

    /// @brief Get CPU elapsed time in milliseconds.
    [[nodiscard]] double cpu_elapsed_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    /// @brief Get effective elapsed time (GPU if available, else CPU).
    [[nodiscard]] double elapsed_ms() const noexcept {
        return gpu_elapsed_ms.has_value() ? static_cast<double>(*gpu_elapsed_ms) : cpu_elapsed_ms();
    }

    /// @brief Get throughput in GB/s (if bytes were recorded).
    [[nodiscard]] std::optional<double> throughput_gbps() const noexcept {
        if (bytes_processed == 0) return std::nullopt;
        double ms = elapsed_ms();
        if (ms <= 0.0) return std::nullopt;
        return static_cast<double>(bytes_processed) / (ms * 1e6); // bytes/ms -> GB/s
    }
};

// ============================================================================
// Scoped Timer
// ============================================================================

class profiler;

/// @brief RAII scoped timer for automatic timing measurement.
///
/// Records timing from construction to destruction. Optionally captures
/// GPU timing when provided with events.
///
/// Example:
/// @code
///   {
///       scoped_timer timer("kernel_launch");
///       // ... work being timed ...
///   } // Timing recorded on destruction
/// @endcode
///
/// @thread_safety Not thread-safe. Use one timer per thread.
class scoped_timer {
public:
    /// @brief Create a standalone timer (no profiler).
    /// @param name Name for this timing region.
    explicit scoped_timer(std::string_view name)
        : name_(name)
        , start_(std::chrono::steady_clock::now())
        , profiler_(nullptr)
        , start_event_(nullptr)
        , end_event_(nullptr)
    {}

    /// @brief Create a timer attached to a profiler.
    /// @param prof The profiler to record to.
    /// @param name Name for this timing region.
    scoped_timer(profiler& prof, std::string_view name);

    /// @brief Create a timer with GPU event timing.
    /// @param name Name for this timing region.
    /// @param start_event GPU event to record at start.
    /// @param end_event GPU event to record at end.
    /// @param q Queue for event recording.
    scoped_timer(std::string_view name, event& start_event, event& end_event, queue& q)
        : name_(name)
        , start_(std::chrono::steady_clock::now())
        , profiler_(nullptr)
        , start_event_(&start_event)
        , end_event_(&end_event)
        , queue_(&q)
    {
        q.record(start_event);
    }

    /// @brief Destructor. Records timing and optionally updates profiler.
    ~scoped_timer();

    // Non-copyable, non-movable
    scoped_timer(const scoped_timer&) = delete;
    scoped_timer& operator=(const scoped_timer&) = delete;
    scoped_timer(scoped_timer&&) = delete;
    scoped_timer& operator=(scoped_timer&&) = delete;

    /// @brief Set the number of bytes processed for throughput calculation.
    void set_bytes(uint64_t bytes) noexcept { bytes_processed_ = bytes; }

    /// @brief Get current elapsed CPU time in milliseconds.
    [[nodiscard]] double elapsed_ms() const noexcept {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    /// @brief Get the timer name.
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    profiler* profiler_;
    event* start_event_;
    event* end_event_;
    queue* queue_{nullptr};
    uint64_t bytes_processed_{0};
};

// ============================================================================
// Event Marker
// ============================================================================

/// @brief GPU timeline marker for debugging and tracing.
///
/// Event markers insert named checkpoints into the GPU command stream,
/// useful for debugging with tools like Nsight Systems or RenderDoc.
///
/// Example:
/// @code
///   event_marker::push(res, "preprocessing");
///   // ... preprocessing work ...
///   event_marker::pop(res);
///
///   // Or use RAII:
///   {
///       event_marker mark(res, "computation");
///       // ... computation work ...
///   } // pop on destruction
/// @endcode
class event_marker {
public:
    /// @brief Push a marker onto the GPU timeline.
    /// @param res GPU resources context.
    /// @param name Marker name.
    static void push(resources& res, std::string_view name);

    /// @brief Pop the most recent marker.
    /// @param res GPU resources context.
    static void pop(resources& res);

    /// @brief Create an RAII marker (pushes on construction, pops on destruction).
    /// @param res GPU resources context.
    /// @param name Marker name.
    event_marker(resources& res, std::string_view name)
        : resources_(&res)
    {
        push(res, name);
    }

    /// @brief Destructor. Pops the marker.
    ~event_marker() {
        if (resources_) {
            pop(*resources_);
        }
    }

    // Non-copyable, movable
    event_marker(const event_marker&) = delete;
    event_marker& operator=(const event_marker&) = delete;

    event_marker(event_marker&& other) noexcept
        : resources_(other.resources_)
    {
        other.resources_ = nullptr;
    }

    event_marker& operator=(event_marker&& other) noexcept {
        if (this != &other) {
            if (resources_) pop(*resources_);
            resources_ = other.resources_;
            other.resources_ = nullptr;
        }
        return *this;
    }

private:
    resources* resources_;
};

// ============================================================================
// Profiler
// ============================================================================

/// @brief Central profiling coordinator for GPU operations.
///
/// Collects timing records from scoped_timer instances and provides
/// aggregated statistics and reports.
///
/// Example:
/// @code
///   navatala::profiling::profiler prof(res);
///
///   for (int i = 0; i < 100; ++i) {
///       auto timer = prof.scope("iteration");
///       // ... work ...
///   }
///
///   // Get statistics
///   auto stats = prof.get_stats("iteration");
///   std::cout << "Mean: " << stats.mean_ms << " ms\n";
///
///   // Print full report
///   prof.report();
/// @endcode
///
/// @thread_safety Thread-safe for recording from multiple threads.
class profiler {
public:
    /// @brief Aggregated statistics for a named timing region.
    struct stats {
        std::string name;
        size_t count{0};
        double total_ms{0.0};
        double min_ms{0.0};
        double max_ms{0.0};
        double mean_ms{0.0};
        double stddev_ms{0.0};
        uint64_t total_bytes{0};
        std::optional<double> mean_throughput_gbps;
    };

    /// @brief Create a profiler attached to GPU resources.
    /// @param res GPU resources context (for event creation).
    explicit profiler(resources& res)
        : resources_(&res)
        , enabled_(true)
    {}

    /// @brief Create a standalone profiler (CPU timing only).
    profiler()
        : resources_(nullptr)
        , enabled_(true)
    {}

    // Non-copyable, non-movable: the profiler owns synchronization state.
    profiler(const profiler&) = delete;
    profiler& operator=(const profiler&) = delete;
    profiler(profiler&&) = delete;
    profiler& operator=(profiler&&) = delete;

    /// @brief Enable or disable profiling.
    /// @param enable True to enable, false to disable.
    void set_enabled(bool enable) noexcept { enabled_ = enable; }

    /// @brief Check if profiling is enabled.
    [[nodiscard]] bool is_enabled() const noexcept { return enabled_; }

    /// @brief Create a scoped timer attached to this profiler.
    /// @param name Name for the timing region.
    /// @return A scoped_timer instance.
    [[nodiscard]] scoped_timer scope(std::string_view name) {
        return scoped_timer(*this, name);
    }

    /// @brief Record a completed timing.
    /// @param record The timing record to add.
    void record(timing_record&& record);

    /// @brief Get statistics for a named region.
    /// @param name The region name to query.
    /// @return Statistics for the region, or nullopt if not found.
    [[nodiscard]] std::optional<stats> get_stats(std::string_view name) const;

    /// @brief Get all timing records.
    /// @return Vector of all recorded timings.
    [[nodiscard]] std::vector<timing_record> get_records() const;

    /// @brief Get statistics for all regions.
    /// @return Vector of stats for each unique region.
    [[nodiscard]] std::vector<stats> get_all_stats() const;

    /// @brief Clear all recorded timings.
    void clear();

    /// @brief Print a formatted report to stdout.
    void report() const;

    /// @brief Generate a report string.
    /// @return Formatted report string.
    [[nodiscard]] std::string report_string() const;

    /// @brief Get the associated GPU resources (if any).
    [[nodiscard]] resources* get_resources() const noexcept { return resources_; }

private:
    resources* resources_;
    std::vector<timing_record> records_;
    mutable std::mutex mutex_;
    std::atomic<bool> enabled_;
};

// ============================================================================
// Free Functions
// ============================================================================

/// @brief Calculate elapsed time between two GPU events.
/// @param start The start event (must be recorded and completed).
/// @param end The end event (must be recorded and completed).
/// @return Elapsed time in milliseconds, or error.
[[nodiscard]] result<float> elapsed_ms(const event& start, const event& end);

/// @brief Calculate elapsed time, throwing on error.
/// @param start The start event.
/// @param end The end event.
/// @return Elapsed time in milliseconds.
/// @throws exception on GPU error.
float elapsed_ms_or_throw(const event& start, const event& end);

/// @brief Measure GPU operation time with automatic event management.
/// @param res GPU resources.
/// @param q Queue to use.
/// @param operation The operation to time (callable).
/// @return Elapsed GPU time in milliseconds, or error.
template<typename Callable>
[[nodiscard]] result<float> measure_gpu(resources& res, queue& q, Callable&& operation) {
    auto start_result = event::create(res);
    if (!start_result.has_value()) {
        return result<float>(start_result.error(), "failed to create start event");
    }

    auto end_result = event::create(res);
    if (!end_result.has_value()) {
        return result<float>(end_result.error(), "failed to create end event");
    }

    auto& start = start_result.value();
    auto& end = end_result.value();

    q.record(start);
    std::forward<Callable>(operation)();
    q.record(end);
    q.sync();

    return elapsed_ms(start, end);
}

/// @brief Measure operation with multiple iterations for averaging.
/// @param res GPU resources.
/// @param q Queue to use.
/// @param iterations Number of iterations.
/// @param warmup_iterations Number of warmup iterations (not timed).
/// @param operation The operation to time.
/// @return Average elapsed time in milliseconds, or error.
template<typename Callable>
[[nodiscard]] result<float> benchmark_gpu(
    resources& res,
    queue& q,
    size_t iterations,
    size_t warmup_iterations,
    Callable&& operation)
{
    // Warmup
    for (size_t i = 0; i < warmup_iterations; ++i) {
        operation();
    }
    q.sync();

    // Create events
    auto start_result = event::create(res);
    if (!start_result.has_value()) {
        return result<float>(start_result.error(), "failed to create start event");
    }

    auto end_result = event::create(res);
    if (!end_result.has_value()) {
        return result<float>(end_result.error(), "failed to create end event");
    }

    auto& start = start_result.value();
    auto& end = end_result.value();

    // Benchmark
    q.record(start);
    for (size_t i = 0; i < iterations; ++i) {
        operation();
    }
    q.record(end);
    q.sync();

    auto elapsed = elapsed_ms(start, end);
    if (!elapsed.has_value()) {
        return elapsed;
    }

    return elapsed.value() / static_cast<float>(iterations);
}

// ============================================================================
// Inline Implementations
// ============================================================================

inline scoped_timer::scoped_timer(profiler& prof, std::string_view name)
    : name_(name)
    , start_(std::chrono::steady_clock::now())
    , profiler_(&prof)
    , start_event_(nullptr)
    , end_event_(nullptr)
{}

inline scoped_timer::~scoped_timer() {
    auto end = std::chrono::steady_clock::now();

    timing_record rec;
    rec.name = name_;
    rec.start_time = start_;
    rec.end_time = end;
    rec.bytes_processed = bytes_processed_;

    // If we have GPU events, record the end and get GPU timing
    if (start_event_ && end_event_ && queue_) {
        queue_->record(*end_event_);
        queue_->sync();
        auto gpu_time = navatala::profiling::elapsed_ms(*start_event_, *end_event_);
        if (gpu_time.has_value()) {
            rec.gpu_elapsed_ms = gpu_time.value();
        }
    }

    // Record to profiler if attached
    if (profiler_ && profiler_->is_enabled()) {
        profiler_->record(std::move(rec));
    }
}

inline void event_marker::push(resources& res, std::string_view name) {
    // Stub implementation - would call into FFI for actual GPU marker push
    // In real implementation this would use:
    // - nvtxRangePush for CUDA
    // - roctxRangePush for HIP
    // - MTLCommandBuffer label for Metal
    (void)res;
    (void)name;
}

inline void event_marker::pop(resources& res) {
    // Stub implementation - would call into FFI for actual GPU marker pop
    (void)res;
}

inline void profiler::record(timing_record&& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back(std::move(record));
}

inline std::optional<profiler::stats> profiler::get_stats(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    stats s;
    s.name = std::string(name);
    s.min_ms = std::numeric_limits<double>::max();
    s.max_ms = 0.0;

    double sum_sq = 0.0;

    for (const auto& rec : records_) {
        if (rec.name == name) {
            double t = rec.elapsed_ms();
            s.count++;
            s.total_ms += t;
            s.total_bytes += rec.bytes_processed;
            s.min_ms = std::min(s.min_ms, t);
            s.max_ms = std::max(s.max_ms, t);
            sum_sq += t * t;
        }
    }

    if (s.count == 0) {
        return std::nullopt;
    }

    s.mean_ms = s.total_ms / static_cast<double>(s.count);

    if (s.count > 1) {
        double variance = (sum_sq / s.count) - (s.mean_ms * s.mean_ms);
        s.stddev_ms = std::sqrt(std::max(0.0, variance));
    }

    if (s.total_bytes > 0 && s.total_ms > 0.0) {
        s.mean_throughput_gbps = static_cast<double>(s.total_bytes) / (s.total_ms * 1e6);
    }

    return s;
}

inline std::vector<timing_record> profiler::get_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

inline std::vector<profiler::stats> profiler::get_all_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Collect unique names
    std::vector<std::string> names;
    for (const auto& rec : records_) {
        bool found = false;
        for (const auto& n : names) {
            if (n == rec.name) {
                found = true;
                break;
            }
        }
        if (!found) {
            names.push_back(rec.name);
        }
    }

    // Compute stats for each - unlock before calling get_stats
    std::vector<stats> result;
    // Note: We can't call get_stats here as it would deadlock
    // So we compute inline
    for (const auto& name : names) {
        stats s;
        s.name = name;
        s.min_ms = std::numeric_limits<double>::max();
        s.max_ms = 0.0;
        double sum_sq = 0.0;

        for (const auto& rec : records_) {
            if (rec.name == name) {
                double t = rec.elapsed_ms();
                s.count++;
                s.total_ms += t;
                s.total_bytes += rec.bytes_processed;
                s.min_ms = std::min(s.min_ms, t);
                s.max_ms = std::max(s.max_ms, t);
                sum_sq += t * t;
            }
        }

        if (s.count > 0) {
            s.mean_ms = s.total_ms / static_cast<double>(s.count);
            if (s.count > 1) {
                double variance = (sum_sq / s.count) - (s.mean_ms * s.mean_ms);
                s.stddev_ms = std::sqrt(std::max(0.0, variance));
            }
            if (s.total_bytes > 0 && s.total_ms > 0.0) {
                s.mean_throughput_gbps = static_cast<double>(s.total_bytes) / (s.total_ms * 1e6);
            }
            result.push_back(std::move(s));
        }
    }

    return result;
}

inline void profiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
}

inline void profiler::report() const {
    // Print to stdout
    auto str = report_string();
    std::printf("%s", str.c_str());
}

inline std::string profiler::report_string() const {
    std::string result;
    result.reserve(1024);

    result += "=== Navatala Profiler Report ===\n\n";

    auto all_stats = get_all_stats();
    if (all_stats.empty()) {
        result += "No timing data recorded.\n";
        return result;
    }

    result += "Region                          Count      Mean       Min        Max       Stddev\n";
    result += "--------------------------------------------------------------------------------\n";

    char buf[256];
    for (const auto& s : all_stats) {
        std::snprintf(buf, sizeof(buf), "%-30s %6zu  %8.3f ms %8.3f ms %8.3f ms %8.3f ms\n",
                      s.name.c_str(), s.count, s.mean_ms, s.min_ms, s.max_ms, s.stddev_ms);
        result += buf;

        if (s.mean_throughput_gbps.has_value()) {
            std::snprintf(buf, sizeof(buf), "  Throughput: %.2f GB/s\n", *s.mean_throughput_gbps);
            result += buf;
        }
    }

    result += "\n";
    return result;
}

} // namespace profiling
} // namespace navatala

// ============================================================================
// Convenience Macros
// ============================================================================

/// @def NAVATALA_PROFILE_SCOPE
/// @brief Macro for easy scoped profiling (when NAVATALA_ENABLE_PROFILING is set).
#if defined(NAVATALA_ENABLE_PROFILING) && NAVATALA_ENABLE_PROFILING
    #define NAVATALA_PROFILE_SCOPE(profiler, name) \
        auto NAVATALA_CONCAT(_navatala_timer_, __LINE__) = (profiler).scope(name)
    #define NAVATALA_PROFILE_GPU(name, start_evt, end_evt, q) \
        navatala::profiling::scoped_timer NAVATALA_CONCAT(_navatala_timer_, __LINE__)(name, start_evt, end_evt, q)
#else
    #define NAVATALA_PROFILE_SCOPE(profiler, name) ((void)0)
    #define NAVATALA_PROFILE_GPU(name, start_evt, end_evt, q) ((void)0)
#endif

// Helper macro for unique variable names
#define NAVATALA_CONCAT_IMPL(a, b) a##b
#define NAVATALA_CONCAT(a, b) NAVATALA_CONCAT_IMPL(a, b)
