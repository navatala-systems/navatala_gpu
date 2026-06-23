// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Worked example: use the public Navatala C++ wrapper API to compute
// y = alpha * x + y. This demonstrates the stable facade layer over the
// lower-level C ABI and generated kernels.

#include "navatala/backend.hpp"
#include "navatala/buffer.hpp"
#include "navatala/linalg.hpp"
#include "navatala/resource.hpp"
#include "navatala/types.hpp"

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace {

bool report_error(const navatala::status& status, const char* label) {
    if (status) {
        return false;
    }
    std::fprintf(stderr, "ERROR: %s failed: %s\n",
                 label, navatala::error_code_name(status.error()));
    return true;
}

template <typename T>
bool report_result_error(const navatala::result<T>& result, const char* label) {
    if (result) {
        return false;
    }
    std::fprintf(stderr, "ERROR: %s failed: %s\n",
                 label, navatala::error_code_name(result.error()));
    return true;
}

}  // namespace

int main() {
    constexpr std::size_t N = 1024;
    constexpr float kAlpha = 2.0f;

    auto resources = navatala::resources::create();
    if (report_result_error(resources, "resources::create")) {
        std::printf("[skip] no Navatala GPU backend/context available\n");
        return 0;
    }

    std::vector<float> host_x(N);
    std::vector<float> host_y(N, 1.0f);
    for (std::size_t i = 0; i < N; ++i) {
        host_x[i] = static_cast<float>(i);
    }

    auto x = navatala::buffer::create(
        resources.value(),
        std::span<const float>(host_x),
        {N},
        navatala::data_type::float32);
    if (report_result_error(x, "buffer::create(x)")) {
        return 1;
    }

    auto y = navatala::buffer::create(
        resources.value(),
        std::span<const float>(host_y),
        {N},
        navatala::data_type::float32);
    if (report_result_error(y, "buffer::create(y)")) {
        return 1;
    }

    auto status = navatala::linalg::axpy(
        x.value(), y.value(), kAlpha, N, resources.value().default_queue());
    if (report_error(status, "linalg::axpy")) {
        return 1;
    }
    resources.value().sync();

    y.value().copy_to_host(std::span<float>(host_y), resources.value().default_queue());
    resources.value().sync();

    int errors = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const float expected = 1.0f + kAlpha * static_cast<float>(i);
        if (std::fabs(host_y[i] - expected) > 1.0e-5f) {
            if (errors < 8) {
                std::fprintf(stderr,
                             "MISMATCH at %zu: got %g, expected %g\n",
                             i,
                             static_cast<double>(host_y[i]),
                             static_cast<double>(expected));
            }
            ++errors;
        }
    }

    if (errors != 0) {
        std::fprintf(stderr, "FAIL: %d / %zu mismatches\n", errors, N);
        return 1;
    }

    std::printf("PASS: wrapper linalg::axpy verified on %s for N=%zu\n",
                navatala::backend_name(resources.value().get_backend()), N);
    return 0;
}
