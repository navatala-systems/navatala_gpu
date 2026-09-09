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

#include <navatala/navatala_ffi.h>
#include <array>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <iostream>

template<class T> T* unallocated() {
    return reinterpret_cast<T*>(std::uintptr_t{1});
}

int main() {
    int checks = 0;
    auto unsupported = [&](NavatalaErrorCode status) {
        assert(status == NAVATALA_NOT_IMPLEMENTED);
        ++checks;
    };
    auto* ctx = unallocated<NavatalaGpuContext>();
    auto* queue = unallocated<NavatalaGpuQueue>();
    auto* fno = unallocated<NavatalaFnoLayer>();
    auto* pino = unallocated<NavatalaPINOLayer>();
    auto* mgn = unallocated<NavatalaMeshGraphNet>();
    auto* event = unallocated<NavatalaGpuEvent>();
    NavatalaFnoConfig fno_config{};
    NavatalaPINOConfig pino_config{};
    NavatalaMGNConfig mgn_config{};
    std::array<float, 8> output{1,2,3,4,5,6,7,8};
    const auto before = output;
    size_t count = 917;
    unsupported(navatala_fno_create(ctx, &fno_config, &fno));
    unsupported(navatala_fno_destroy(fno));
    unsupported(navatala_fno_forward(fno, output.data(), 1, output.data(), queue));
    unsupported(navatala_fno_forward_with_params(fno, output.data(), output.data(), 1, output.data(), queue));
    unsupported(navatala_fno_save(fno, ""));
    unsupported(navatala_fno_load(ctx, "", &fno_config, &fno));
    unsupported(navatala_fno_num_params(fno, &count));
    unsupported(navatala_pino_create(ctx, &pino_config, &pino));
    unsupported(navatala_pino_destroy(pino));
    unsupported(navatala_pino_forward(pino, output.data(), 1, output.data(), queue));
    unsupported(navatala_pino_train_step(pino, output.data(), output.data(), 1,
        &output[0], &output[1], &output[2], &output[3], queue));
    unsupported(navatala_pino_compute_residual(pino, output.data(), 1, &output[4], queue));
    unsupported(navatala_pino_save(pino, ""));
    unsupported(navatala_pino_load(ctx, "", &pino_config, &pino));
    unsupported(navatala_mgn_create(ctx, &mgn_config, &mgn));
    unsupported(navatala_mgn_destroy(mgn));
    unsupported(navatala_mgn_forward(mgn, output.data(), 1, output.data(), 1, output.data(), output.data(), queue));
    unsupported(navatala_mgn_forward_no_edge_features(mgn, output.data(), 1, output.data(), 1, output.data(), queue));
    unsupported(navatala_mgn_build_edge_index(ctx, output.data(), 1, 3, output.data(), &count, queue));
    unsupported(navatala_mgn_compute_edge_features(ctx, output.data(), 1, output.data(), 1, output.data(), queue));
    unsupported(navatala_mgn_save(mgn, ""));
    unsupported(navatala_mgn_load(ctx, "", &mgn_config, &mgn));
    unsupported(navatala_mgn_num_params(mgn, &count));
    unsupported(navatala_event_elapsed_ms(event, event, &output[5]));
    unsupported(navatala_profiler_push_marker(ctx, "marker"));
    unsupported(navatala_profiler_pop_marker(ctx));
    auto* index = unallocated<NavatalaVectorIndex>();
    NavatalaCagraBuildParams cagra{};
    NavatalaCagraSearchParams cagra_search{};
    NavatalaIvfpqBuildParams pq{};
    NavatalaIvfpqSearchParams pq_search{};
    NavatalaIvfflatBuildParams flat{};
    NavatalaIvfflatSearchParams flat_search{};
    int metric = 7;
    unsupported(navatala_cagra_build(ctx, output.data(), 1, 1, &cagra, &index));
    unsupported(navatala_cagra_load(ctx, "", &index, &count, &count, &count, &metric));
    unsupported(navatala_cagra_search(index, output.data(), 1, 1, &cagra_search, output.data(), output.data(), queue));
    unsupported(navatala_cagra_save(index, ""));
    unsupported(navatala_ivfpq_build(ctx, output.data(), 1, 1, &pq, &index));
    unsupported(navatala_ivfpq_load(ctx, "", &index, &count, &count, &count, &count, &count, &metric));
    unsupported(navatala_ivfpq_search(index, output.data(), 1, 1, &pq_search, output.data(), output.data(), queue));
    unsupported(navatala_ivfpq_save(index, ""));
    unsupported(navatala_ivfflat_build(ctx, output.data(), 1, 1, &flat, &index));
    unsupported(navatala_ivfflat_load(ctx, "", &index, &count, &count, &count, &metric));
    unsupported(navatala_ivfflat_search(index, output.data(), 1, 1, &flat_search, output.data(), output.data(), queue));
    unsupported(navatala_ivfflat_save(index, ""));
    unsupported(navatala_brute_force_search(ctx, output.data(), 1, 1, output.data(), 1, 1, 0, output.data(), output.data(), queue));
    unsupported(navatala_pairwise_distance(output.data(), 1, output.data(), 1, 1, 0, output.data(), queue));
    unsupported(navatala_fused_l2_nn(output.data(), 1, output.data(), 1, 1, output.data(), output.data(), queue));
    unsupported(navatala_distance_row_min(output.data(), 1, 1, output.data(), queue));
    unsupported(navatala_distance_row_argmin(output.data(), 1, 1, output.data(), queue));
    unsupported(navatala_distance_row_topk(output.data(), 1, 1, 1, 0, output.data(), output.data(), queue));
    unsupported(navatala_normalize_l2(output.data(), 1, 1, queue));
    unsupported(navatala_compute_norms(output.data(), 1, 1, output.data(), queue));
    unsupported(navatala_kmeans_fit(ctx, output.data(), 1, 1, 1, 1, .01f, 0, 0, 0,
        output.data(), &output[6], &count, queue));
    unsupported(navatala_kmeans_predict(ctx, output.data(), 1, 1, output.data(), 1, 0, output.data(), queue));
    unsupported(navatala_kmeans_transform(ctx, output.data(), 1, 1, output.data(), 1, 0, output.data(), queue));
    assert(index == unallocated<NavatalaVectorIndex>() && metric == 7);
    assert(fno == unallocated<NavatalaFnoLayer>());
    assert(pino == unallocated<NavatalaPINOLayer>());
    assert(mgn == unallocated<NavatalaMeshGraphNet>());
    assert(output == before && count == 917);
    std::cout << checks << " unsupported calls preserved all outputs\n";
}
