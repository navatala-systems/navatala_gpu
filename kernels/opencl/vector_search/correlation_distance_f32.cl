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

__kernel void navatala_vector_search_correlation_distance_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
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
    float sum_x = as_float(0x00000000u);
    float sum_y = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float old_sx = sum_x;
      float old_sy = sum_y;
      sum_x = (old_sx + x_val);
      sum_y = (old_sy + y_val);
    }
    float d_f = ((float)(d));
    float final_sx = sum_x;
    float final_sy = sum_y;
    float mean_x = (final_sx / d_f);
    float mean_y = (final_sy / d_f);
    float cov = as_float(0x00000000u);
    float var_x = as_float(0x00000000u);
    float var_y = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx2 = ((gid_x * d) + k);
      uint y_idx2 = ((gid_y * d) + k);
      float x_val2 = x[x_idx2];
      float y_val2 = y[y_idx2];
      float dx = (x_val2 - mean_x);
      float dy = (y_val2 - mean_y);
      float old_cov = cov;
      float old_vx = var_x;
      float old_vy = var_y;
      cov = (old_cov + (dx * dy));
      var_x = (old_vx + (dx * dx));
      var_y = (old_vy + (dy * dy));
    }
    float final_cov = cov;
    float final_vx = var_x;
    float final_vy = var_y;
    float var_prod = (final_vx * final_vy);
    float sqrt_var = sqrt(var_prod);
    float eps = as_float(0x322bcc77u);
    float safe_denom = (sqrt_var + eps);
    float correlation = (final_cov / safe_denom);
    float corr_dist = (as_float(0x3f800000u) - correlation);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = corr_dist;
  }
}
