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

// Navatala C++ Wrapper Layer - Profiling Implementation
// Part of the Navatala GPU Runtime Library
// Polish

#include "../include/navatala/profiling.hpp"
#include "../include/navatala/navatala_ffi.h"

namespace navatala {
namespace profiling {

// ============================================================================
// Free Function Implementations
// ============================================================================

result<float> elapsed_ms(const event& start, const event& end) {
    if (!start.is_valid() || !end.is_valid()) {
        return result<float>(error_code::invalid_handle, "invalid event handle");
    }

    float ms = 0.0f;
    NavatalaErrorCode err = navatala_event_elapsed_ms(
        start.handle(),
        end.handle(),
        &ms
    );

    if (err != NAVATALA_SUCCESS) {
        return result<float>(translate_error(err), "failed to get elapsed time");
    }

    return ms;
}

float elapsed_ms_or_throw(const event& start, const event& end) {
    auto result = elapsed_ms(start, end);
    if (!result.has_value()) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(result.error(), "failed to get elapsed time");
#else
        return 0.0f;
#endif
    }
    return result.value();
}

} // namespace profiling
} // namespace navatala
