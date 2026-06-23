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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_vector_search_reorder_vertices_by_degree(const unsigned int* degrees, unsigned int* prefix_sums, const unsigned int* n_vertices, const unsigned int* n_bins, unsigned int* old_to_new, unsigned int* new_to_old) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int nb = n_bins[0];
  if (vid < nv) {
    unsigned int deg = degrees[vid];
    unsigned int bin = (((deg >= nb)) ? ((nb - 1u)) : (deg));
    unsigned int new_id = atomicAdd(&(prefix_sums[bin]), 1u);
    old_to_new[vid] = new_id;
    new_to_old[new_id] = vid;
  }
}
