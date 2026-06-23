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

// Navatala C++ Wrapper Layer - Runtime Metrics Collection
// Part of the Navatala GPU Runtime Library
// Polish

#pragma once

#include "config.hpp"
#include "error.hpp"

#include <chrono>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <memory>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <unordered_map>

/// @file navatala/metrics.hpp
/// @brief Runtime statistics and metrics collection for GPU operations.
///
/// This header provides:
/// - counter: Thread-safe atomic counter for operation counts
/// - histogram: Latency distribution tracking
/// - gauge: Current value tracking (e.g., memory usage)
/// - throughput_meter: Bytes/operations per second tracking
/// - metrics_registry: Central metrics coordinator
///
/// @section metrics_usage Usage
///
/// @code
///   navatala::metrics::metrics_registry registry;
///
///   // Create metrics
///   auto& ops = registry.counter("gpu_operations");
///   auto& latency = registry.histogram("kernel_latency_ms");
///   auto& mem_usage = registry.gauge("memory_used_bytes");
///
///   // Record metrics
///   ops.increment();
///   latency.observe(kernel_time_ms);
///   mem_usage.set(current_bytes);
///
///   // Report
///   registry.report();
/// @endcode

namespace navatala {
namespace metrics {

// ============================================================================
// Counter
// ============================================================================

/// @brief Thread-safe atomic counter for counting events.
///
/// Counters only go up (or can be reset to zero). Use for counting
/// operations, allocations, cache hits, etc.
///
/// @thread_safety All operations are thread-safe.
class counter {
public:
    /// @brief Create a counter with the given name.
    /// @param name Metric name.
    explicit counter(std::string_view name)
        : name_(name)
        , value_(0)
    {}

    /// @brief Increment the counter by 1.
    void increment() noexcept {
        value_.fetch_add(1, std::memory_order_relaxed);
    }

    /// @brief Add a value to the counter.
    /// @param delta Amount to add.
    void add(uint64_t delta) noexcept {
        value_.fetch_add(delta, std::memory_order_relaxed);
    }

    /// @brief Get the current counter value.
    [[nodiscard]] uint64_t value() const noexcept {
        return value_.load(std::memory_order_relaxed);
    }

    /// @brief Reset the counter to zero.
    void reset() noexcept {
        value_.store(0, std::memory_order_relaxed);
    }

    /// @brief Get the counter name.
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

private:
    std::string name_;
    std::atomic<uint64_t> value_;
};

// ============================================================================
// Gauge
// ============================================================================

/// @brief Thread-safe gauge for tracking current values.
///
/// Gauges can go up or down. Use for memory usage, queue depth,
/// active connections, etc.
///
/// @thread_safety All operations are thread-safe.
class gauge {
public:
    /// @brief Create a gauge with the given name.
    /// @param name Metric name.
    explicit gauge(std::string_view name)
        : name_(name)
        , value_(0)
    {}

    /// @brief Set the gauge value.
    /// @param val New value.
    void set(int64_t val) noexcept {
        value_.store(val, std::memory_order_relaxed);
    }

    /// @brief Increment the gauge by 1.
    void increment() noexcept {
        value_.fetch_add(1, std::memory_order_relaxed);
    }

    /// @brief Decrement the gauge by 1.
    void decrement() noexcept {
        value_.fetch_sub(1, std::memory_order_relaxed);
    }

    /// @brief Add a value to the gauge.
    /// @param delta Amount to add (can be negative).
    void add(int64_t delta) noexcept {
        value_.fetch_add(delta, std::memory_order_relaxed);
    }

    /// @brief Get the current gauge value.
    [[nodiscard]] int64_t value() const noexcept {
        return value_.load(std::memory_order_relaxed);
    }

    /// @brief Get the gauge name.
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

private:
    std::string name_;
    std::atomic<int64_t> value_;
};

// ============================================================================
// Histogram
// ============================================================================

/// @brief Thread-safe histogram for distribution tracking.
///
/// Tracks the distribution of values (e.g., latencies) using fixed buckets.
/// Provides percentiles, mean, and standard deviation.
///
/// @thread_safety All operations are thread-safe.
class histogram {
public:
    /// @brief Default bucket boundaries (in milliseconds).
    static constexpr std::array<double, 16> default_buckets = {
        0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0,
        50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0
    };

    /// @brief Histogram statistics.
    struct stats {
        size_t count{0};
        double sum{0.0};
        double min{0.0};
        double max{0.0};
        double mean{0.0};
        double stddev{0.0};
        double p50{0.0};  ///< 50th percentile (median)
        double p90{0.0};  ///< 90th percentile
        double p95{0.0};  ///< 95th percentile
        double p99{0.0};  ///< 99th percentile
    };

    /// @brief Create a histogram with the given name and default buckets.
    /// @param name Metric name.
    explicit histogram(std::string_view name)
        : name_(name)
        , bucket_boundaries_(default_buckets.begin(), default_buckets.end())
        , buckets_(default_buckets.size() + 1, 0)  // +1 for overflow bucket
        , count_(0)
        , sum_(0.0)
        , sum_sq_(0.0)
        , min_(std::numeric_limits<double>::max())
        , max_(std::numeric_limits<double>::lowest())
    {}

    /// @brief Create a histogram with custom bucket boundaries.
    /// @param name Metric name.
    /// @param boundaries Sorted bucket boundaries.
    histogram(std::string_view name, std::vector<double> boundaries)
        : name_(name)
        , bucket_boundaries_(std::move(boundaries))
        , buckets_(bucket_boundaries_.size() + 1, 0)
        , count_(0)
        , sum_(0.0)
        , sum_sq_(0.0)
        , min_(std::numeric_limits<double>::max())
        , max_(std::numeric_limits<double>::lowest())
    {}

    /// @brief Record an observation.
    /// @param value The observed value.
    void observe(double value) {
        std::lock_guard<std::mutex> lock(mutex_);

        count_++;
        sum_ += value;
        sum_sq_ += value * value;
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);

        // Find bucket
        size_t bucket = bucket_boundaries_.size(); // overflow by default
        for (size_t i = 0; i < bucket_boundaries_.size(); ++i) {
            if (value <= bucket_boundaries_[i]) {
                bucket = i;
                break;
            }
        }
        buckets_[bucket]++;
    }

    /// @brief Get computed statistics.
    [[nodiscard]] stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        stats s;
        s.count = count_;
        s.sum = sum_;
        s.min = (count_ > 0) ? min_ : 0.0;
        s.max = (count_ > 0) ? max_ : 0.0;

        if (count_ > 0) {
            s.mean = sum_ / static_cast<double>(count_);
            if (count_ > 1) {
                double variance = (sum_sq_ / count_) - (s.mean * s.mean);
                s.stddev = std::sqrt(std::max(0.0, variance));
            }

            // Compute percentiles from buckets
            s.p50 = percentile_locked(0.50);
            s.p90 = percentile_locked(0.90);
            s.p95 = percentile_locked(0.95);
            s.p99 = percentile_locked(0.99);
        }

        return s;
    }

    /// @brief Reset the histogram.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::fill(buckets_.begin(), buckets_.end(), 0);
        count_ = 0;
        sum_ = 0.0;
        sum_sq_ = 0.0;
        min_ = std::numeric_limits<double>::max();
        max_ = std::numeric_limits<double>::lowest();
    }

    /// @brief Get the histogram name.
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

    /// @brief Get bucket boundaries.
    [[nodiscard]] const std::vector<double>& bucket_boundaries() const noexcept {
        return bucket_boundaries_;
    }

    /// @brief Get bucket counts.
    [[nodiscard]] std::vector<size_t> bucket_counts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buckets_;
    }

private:
    // Must be called with mutex held
    [[nodiscard]] double percentile_locked(double p) const {
        if (count_ == 0) return 0.0;

        size_t target = static_cast<size_t>(p * count_);
        size_t cumulative = 0;

        for (size_t i = 0; i < buckets_.size(); ++i) {
            cumulative += buckets_[i];
            if (cumulative >= target) {
                // Linear interpolation within bucket
                if (i == 0) {
                    return bucket_boundaries_[0];
                } else if (i >= bucket_boundaries_.size()) {
                    return max_; // Overflow bucket
                } else {
                    double lower = bucket_boundaries_[i - 1];
                    double upper = bucket_boundaries_[i];
                    return (lower + upper) / 2.0;
                }
            }
        }

        return max_;
    }

    std::string name_;
    std::vector<double> bucket_boundaries_;
    std::vector<size_t> buckets_;
    size_t count_;
    double sum_;
    double sum_sq_;
    double min_;
    double max_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Throughput Meter
// ============================================================================

/// @brief Tracks throughput (bytes or operations per second).
///
/// Computes rolling average throughput over a time window.
///
/// @thread_safety All operations are thread-safe.
class throughput_meter {
public:
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;

    /// @brief Create a throughput meter.
    /// @param name Metric name.
    /// @param window_seconds Window size for averaging.
    explicit throughput_meter(std::string_view name, double window_seconds = 5.0)
        : name_(name)
        , window_ns_(static_cast<int64_t>(window_seconds * 1e9))
        , total_bytes_(0)
        , total_ops_(0)
        , start_time_(clock::now())
    {}

    /// @brief Record bytes transferred.
    /// @param bytes Number of bytes.
    void record_bytes(uint64_t bytes) noexcept {
        total_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }

    /// @brief Record an operation.
    void record_op() noexcept {
        total_ops_.fetch_add(1, std::memory_order_relaxed);
    }

    /// @brief Record bytes and operation together.
    /// @param bytes Number of bytes.
    void record(uint64_t bytes) noexcept {
        record_bytes(bytes);
        record_op();
    }

    /// @brief Get current throughput in bytes per second.
    [[nodiscard]] double bytes_per_second() const noexcept {
        auto now = clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed <= 0.0) return 0.0;
        return static_cast<double>(total_bytes_.load(std::memory_order_relaxed)) / elapsed;
    }

    /// @brief Get current throughput in GB/s.
    [[nodiscard]] double gbps() const noexcept {
        return bytes_per_second() / 1e9;
    }

    /// @brief Get operations per second.
    [[nodiscard]] double ops_per_second() const noexcept {
        auto now = clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed <= 0.0) return 0.0;
        return static_cast<double>(total_ops_.load(std::memory_order_relaxed)) / elapsed;
    }

    /// @brief Get total bytes recorded.
    [[nodiscard]] uint64_t total_bytes() const noexcept {
        return total_bytes_.load(std::memory_order_relaxed);
    }

    /// @brief Get total operations recorded.
    [[nodiscard]] uint64_t total_ops() const noexcept {
        return total_ops_.load(std::memory_order_relaxed);
    }

    /// @brief Get elapsed time in seconds.
    [[nodiscard]] double elapsed_seconds() const noexcept {
        auto now = clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    /// @brief Reset the meter.
    void reset() noexcept {
        total_bytes_.store(0, std::memory_order_relaxed);
        total_ops_.store(0, std::memory_order_relaxed);
        start_time_ = clock::now();
    }

    /// @brief Get the meter name.
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

private:
    std::string name_;
    int64_t window_ns_;
    std::atomic<uint64_t> total_bytes_;
    std::atomic<uint64_t> total_ops_;
    time_point start_time_;
};

// ============================================================================
// Memory Stats
// ============================================================================

/// @brief Memory usage statistics.
struct memory_stats {
    uint64_t allocated_bytes{0};      ///< Currently allocated bytes
    uint64_t peak_allocated_bytes{0}; ///< Peak allocation
    uint64_t total_allocated{0};      ///< Cumulative allocations
    uint64_t total_deallocated{0};    ///< Cumulative deallocations
    uint64_t allocation_count{0};     ///< Number of allocations
    uint64_t deallocation_count{0};   ///< Number of deallocations
    uint64_t pool_capacity{0};        ///< Memory pool capacity
    uint64_t pool_free{0};            ///< Available pool memory
};

/// @brief Thread-safe memory statistics tracker.
class memory_tracker {
public:
    /// @brief Create a memory tracker.
    /// @param name Tracker name (e.g., "gpu_device_memory").
    explicit memory_tracker(std::string_view name)
        : name_(name)
        , allocated_(0)
        , peak_(0)
        , total_allocated_(0)
        , total_deallocated_(0)
        , alloc_count_(0)
        , dealloc_count_(0)
    {}

    /// @brief Record an allocation.
    /// @param bytes Number of bytes allocated.
    void record_alloc(uint64_t bytes) noexcept {
        uint64_t current = allocated_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        alloc_count_.fetch_add(1, std::memory_order_relaxed);

        // Update peak (relaxed CAS loop)
        uint64_t peak = peak_.load(std::memory_order_relaxed);
        while (current > peak) {
            if (peak_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
                break;
            }
        }
    }

    /// @brief Record a deallocation.
    /// @param bytes Number of bytes deallocated.
    void record_dealloc(uint64_t bytes) noexcept {
        allocated_.fetch_sub(bytes, std::memory_order_relaxed);
        total_deallocated_.fetch_add(bytes, std::memory_order_relaxed);
        dealloc_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// @brief Get current statistics.
    [[nodiscard]] memory_stats get_stats() const noexcept {
        memory_stats s;
        s.allocated_bytes = allocated_.load(std::memory_order_relaxed);
        s.peak_allocated_bytes = peak_.load(std::memory_order_relaxed);
        s.total_allocated = total_allocated_.load(std::memory_order_relaxed);
        s.total_deallocated = total_deallocated_.load(std::memory_order_relaxed);
        s.allocation_count = alloc_count_.load(std::memory_order_relaxed);
        s.deallocation_count = dealloc_count_.load(std::memory_order_relaxed);
        return s;
    }

    /// @brief Get currently allocated bytes.
    [[nodiscard]] uint64_t allocated_bytes() const noexcept {
        return allocated_.load(std::memory_order_relaxed);
    }

    /// @brief Get peak allocated bytes.
    [[nodiscard]] uint64_t peak_bytes() const noexcept {
        return peak_.load(std::memory_order_relaxed);
    }

    /// @brief Reset statistics.
    void reset() noexcept {
        allocated_.store(0, std::memory_order_relaxed);
        peak_.store(0, std::memory_order_relaxed);
        total_allocated_.store(0, std::memory_order_relaxed);
        total_deallocated_.store(0, std::memory_order_relaxed);
        alloc_count_.store(0, std::memory_order_relaxed);
        dealloc_count_.store(0, std::memory_order_relaxed);
    }

    /// @brief Get the tracker name.
    [[nodiscard]] std::string_view name() const noexcept { return name_; }

private:
    std::string name_;
    std::atomic<uint64_t> allocated_;
    std::atomic<uint64_t> peak_;
    std::atomic<uint64_t> total_allocated_;
    std::atomic<uint64_t> total_deallocated_;
    std::atomic<uint64_t> alloc_count_;
    std::atomic<uint64_t> dealloc_count_;
};

// ============================================================================
// Metrics Registry
// ============================================================================

/// @brief Central registry for all metrics.
///
/// Provides a single point of access for creating and querying metrics.
/// All created metrics are owned by the registry.
///
/// @thread_safety Creating and accessing metrics is thread-safe.
class metrics_registry {
public:
    /// @brief Create or get a counter by name.
    /// @param name Counter name.
    /// @return Reference to the counter.
    counter& get_counter(std::string_view name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = counters_.find(std::string(name));
        if (it != counters_.end()) {
            return *it->second;
        }
        auto [new_it, inserted] = counters_.emplace(
            std::string(name),
            std::make_unique<counter>(name)
        );
        return *new_it->second;
    }

    /// @brief Create or get a gauge by name.
    /// @param name Gauge name.
    /// @return Reference to the gauge.
    gauge& get_gauge(std::string_view name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = gauges_.find(std::string(name));
        if (it != gauges_.end()) {
            return *it->second;
        }
        auto [new_it, inserted] = gauges_.emplace(
            std::string(name),
            std::make_unique<gauge>(name)
        );
        return *new_it->second;
    }

    /// @brief Create or get a histogram by name.
    /// @param name Histogram name.
    /// @return Reference to the histogram.
    histogram& get_histogram(std::string_view name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = histograms_.find(std::string(name));
        if (it != histograms_.end()) {
            return *it->second;
        }
        auto [new_it, inserted] = histograms_.emplace(
            std::string(name),
            std::make_unique<histogram>(name)
        );
        return *new_it->second;
    }

    /// @brief Create or get a throughput meter by name.
    /// @param name Meter name.
    /// @return Reference to the meter.
    throughput_meter& get_throughput_meter(std::string_view name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = meters_.find(std::string(name));
        if (it != meters_.end()) {
            return *it->second;
        }
        auto [new_it, inserted] = meters_.emplace(
            std::string(name),
            std::make_unique<throughput_meter>(name)
        );
        return *new_it->second;
    }

    /// @brief Create or get a memory tracker by name.
    /// @param name Tracker name.
    /// @return Reference to the tracker.
    memory_tracker& get_memory_tracker(std::string_view name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = memory_trackers_.find(std::string(name));
        if (it != memory_trackers_.end()) {
            return *it->second;
        }
        auto [new_it, inserted] = memory_trackers_.emplace(
            std::string(name),
            std::make_unique<memory_tracker>(name)
        );
        return *new_it->second;
    }

    /// @brief Reset all metrics.
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [name, c] : counters_) c->reset();
        for (auto& [name, h] : histograms_) h->reset();
        for (auto& [name, m] : meters_) m->reset();
        for (auto& [name, t] : memory_trackers_) t->reset();
        // Gauges are not reset - they represent current state
    }

    /// @brief Print a formatted report to stdout.
    void report() const {
        std::printf("%s", report_string().c_str());
    }

    /// @brief Generate a report string.
    [[nodiscard]] std::string report_string() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string result;
        result.reserve(4096);

        result += "=== Navatala Metrics Report ===\n\n";

        // Counters
        if (!counters_.empty()) {
            result += "-- Counters --\n";
            for (const auto& [name, c] : counters_) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "  %-40s %lu\n",
                              name.c_str(), static_cast<unsigned long>(c->value()));
                result += buf;
            }
            result += "\n";
        }

        // Gauges
        if (!gauges_.empty()) {
            result += "-- Gauges --\n";
            for (const auto& [name, g] : gauges_) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "  %-40s %ld\n",
                              name.c_str(), static_cast<long>(g->value()));
                result += buf;
            }
            result += "\n";
        }

        // Histograms
        if (!histograms_.empty()) {
            result += "-- Histograms --\n";
            for (const auto& [name, h] : histograms_) {
                auto s = h->get_stats();
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "  %-40s count=%zu mean=%.3f stddev=%.3f p50=%.3f p95=%.3f p99=%.3f\n",
                    name.c_str(), s.count, s.mean, s.stddev, s.p50, s.p95, s.p99);
                result += buf;
            }
            result += "\n";
        }

        // Throughput meters
        if (!meters_.empty()) {
            result += "-- Throughput --\n";
            for (const auto& [name, m] : meters_) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "  %-40s %.2f GB/s  %.0f ops/s  (%.2f s elapsed)\n",
                    name.c_str(), m->gbps(), m->ops_per_second(), m->elapsed_seconds());
                result += buf;
            }
            result += "\n";
        }

        // Memory trackers
        if (!memory_trackers_.empty()) {
            result += "-- Memory --\n";
            for (const auto& [name, t] : memory_trackers_) {
                auto s = t->get_stats();
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "  %-40s current=%.2f MB peak=%.2f MB allocs=%lu\n",
                    name.c_str(),
                    static_cast<double>(s.allocated_bytes) / 1e6,
                    static_cast<double>(s.peak_allocated_bytes) / 1e6,
                    static_cast<unsigned long>(s.allocation_count));
                result += buf;
            }
            result += "\n";
        }

        return result;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<counter>> counters_;
    std::unordered_map<std::string, std::unique_ptr<gauge>> gauges_;
    std::unordered_map<std::string, std::unique_ptr<histogram>> histograms_;
    std::unordered_map<std::string, std::unique_ptr<throughput_meter>> meters_;
    std::unordered_map<std::string, std::unique_ptr<memory_tracker>> memory_trackers_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Global Registry (optional)
// ============================================================================

/// @brief Get the global metrics registry singleton.
/// @return Reference to the global registry.
inline metrics_registry& global_registry() {
    static metrics_registry instance;
    return instance;
}

// ============================================================================
// Convenience Macros
// ============================================================================

/// @def NAVATALA_METRIC_COUNTER
/// @brief Increment a global counter by name.
#if defined(NAVATALA_ENABLE_METRICS) && NAVATALA_ENABLE_METRICS
    #define NAVATALA_METRIC_COUNTER(name) \
        navatala::metrics::global_registry().get_counter(name).increment()
    #define NAVATALA_METRIC_COUNTER_ADD(name, value) \
        navatala::metrics::global_registry().get_counter(name).add(value)
    #define NAVATALA_METRIC_HISTOGRAM(name, value) \
        navatala::metrics::global_registry().get_histogram(name).observe(value)
    #define NAVATALA_METRIC_GAUGE_SET(name, value) \
        navatala::metrics::global_registry().get_gauge(name).set(value)
#else
    #define NAVATALA_METRIC_COUNTER(name) ((void)0)
    #define NAVATALA_METRIC_COUNTER_ADD(name, value) ((void)0)
    #define NAVATALA_METRIC_HISTOGRAM(name, value) ((void)0)
    #define NAVATALA_METRIC_GAUGE_SET(name, value) ((void)0)
#endif

} // namespace metrics
} // namespace navatala
