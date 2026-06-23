// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Worked example: use the public Navatala C++ wrapper API to compute
// y = A * x for a small CSR sparse matrix.

#include "navatala/backend.hpp"
#include "navatala/buffer.hpp"
#include "navatala/linalg.hpp"
#include "navatala/resource.hpp"
#include "navatala/types.hpp"

#include <cmath>
#include <cstdint>
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
    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 3;
    constexpr std::size_t nnz = 5;

    // A = [ 10  0  0 ]
    //     [  0 20  1 ]
    //     [  2  0 30 ]
    // x = [1, 2, 3], so y = [10, 43, 92].
    const std::vector<std::uint32_t> row_offsets = {0, 1, 3, 5};
    const std::vector<std::uint32_t> col_indices = {0, 1, 2, 0, 2};
    const std::vector<float> values = {10.0f, 20.0f, 1.0f, 2.0f, 30.0f};
    const std::vector<float> x_host = {1.0f, 2.0f, 3.0f};
    std::vector<float> y_host(rows, 0.0f);

    auto resources = navatala::resources::create();
    if (report_result_error(resources, "resources::create")) {
        std::printf("[skip] no Navatala GPU backend/context available\n");
        return 0;
    }

    auto row_buffer = navatala::buffer::create(
        resources.value(),
        std::span<const std::uint32_t>(row_offsets),
        {rows + 1},
        navatala::data_type::uint32);
    if (report_result_error(row_buffer, "buffer::create(row_offsets)")) {
        return 1;
    }

    auto col_buffer = navatala::buffer::create(
        resources.value(),
        std::span<const std::uint32_t>(col_indices),
        {nnz},
        navatala::data_type::uint32);
    if (report_result_error(col_buffer, "buffer::create(col_indices)")) {
        return 1;
    }

    auto values_buffer = navatala::buffer::create(
        resources.value(),
        std::span<const float>(values),
        {nnz},
        navatala::data_type::float32);
    if (report_result_error(values_buffer, "buffer::create(values)")) {
        return 1;
    }

    auto x_buffer = navatala::buffer::create(
        resources.value(),
        std::span<const float>(x_host),
        {cols},
        navatala::data_type::float32);
    if (report_result_error(x_buffer, "buffer::create(x)")) {
        return 1;
    }

    auto y_buffer = navatala::buffer::create(
        resources.value(),
        {rows},
        navatala::data_type::float32);
    if (report_result_error(y_buffer, "buffer::create(y)")) {
        return 1;
    }

    auto status = navatala::sparse::csr_spmv(
        row_buffer.value(),
        col_buffer.value(),
        values_buffer.value(),
        x_buffer.value(),
        y_buffer.value(),
        rows,
        cols,
        nnz,
        resources.value().default_queue());
    if (report_error(status, "sparse::csr_spmv")) {
        return 1;
    }
    resources.value().sync();

    y_buffer.value().copy_to_host(
        std::span<float>(y_host), resources.value().default_queue());
    resources.value().sync();

    const std::vector<float> expected = {10.0f, 43.0f, 92.0f};
    int errors = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        if (std::fabs(y_host[i] - expected[i]) > 1.0e-5f) {
            std::fprintf(stderr,
                         "MISMATCH at %zu: got %g, expected %g\n",
                         i,
                         static_cast<double>(y_host[i]),
                         static_cast<double>(expected[i]));
            ++errors;
        }
    }

    if (errors != 0) {
        std::fprintf(stderr, "FAIL: %d / %zu mismatches\n", errors, rows);
        return 1;
    }

    std::printf("PASS: wrapper sparse::csr_spmv verified on %s for %zux%zu, nnz=%zu\n",
                navatala::backend_name(resources.value().get_backend()),
                rows,
                cols,
                nnz);
    return 0;
}
