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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_compute_residuals_f32(const float* vectors, const float* centroids, const unsigned int* assignments, const unsigned int* n_vectors, const unsigned int* dim, float* residuals) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  unsigned int total = (nv * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int vec_idx = (gid / d);
    unsigned int dim_idx = (gid % d);
    unsigned int cluster = assignments[vec_idx];
    float v_val = vectors[gid];
    unsigned int c_idx = ((cluster * d) + dim_idx);
    float c_val = centroids[c_idx];
    float residual = (v_val - c_val);
    residuals[gid] = residual;
  }
}
