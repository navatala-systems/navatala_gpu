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

__kernel void navatala_vector_search_ivfpq_compute_residuals_f32(__global const float* vectors, __global const float* centroids, __global const uint* assignments, __global const uint* n_vectors, __global const uint* dim, __global float* residuals) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint vec_idx = (gid / d);
    uint dim_idx = (gid % d);
    uint cluster = assignments[vec_idx];
    float v_val = vectors[gid];
    uint c_idx = ((cluster * d) + dim_idx);
    float c_val = centroids[c_idx];
    float residual = (v_val - c_val);
    residuals[gid] = residual;
  }
}
