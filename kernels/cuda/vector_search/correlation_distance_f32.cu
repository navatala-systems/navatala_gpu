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
extern "C" __global__ void navatala_vector_search_correlation_distance_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float sum_x = __uint_as_float(0x00000000u);
    float sum_y = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
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
    float cov = __uint_as_float(0x00000000u);
    float var_x = __uint_as_float(0x00000000u);
    float var_y = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx2 = ((gid_x * d) + k);
      unsigned int y_idx2 = ((gid_y * d) + k);
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
    float eps = __uint_as_float(0x322bcc77u);
    float safe_denom = (sqrt_var + eps);
    float correlation = (final_cov / safe_denom);
    float corr_dist = (__uint_as_float(0x3f800000u) - correlation);
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = corr_dist;
  }
}
