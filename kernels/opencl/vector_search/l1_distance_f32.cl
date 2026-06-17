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

__kernel void navatala_vector_search_l1_distance_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float neg_diff = (as_float(0x00000000u) - diff);
      bool is_neg = (diff < as_float(0x00000000u));
      float abs_diff = ((is_neg) ? (neg_diff) : (diff));
      float old_acc = acc;
      acc = (old_acc + abs_diff);
    }
    uint dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}
