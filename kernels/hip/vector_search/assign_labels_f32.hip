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
extern "C" __global__ void navatala_vector_search_assign_labels_f32(const float* data, const float* centroids, const unsigned int* n_points, const unsigned int* k, const unsigned int* dim, unsigned int* labels, float* min_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = n_points[0];
  unsigned int kVal = k[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float best_dist = __uint_as_float(0x7e967699u);
    unsigned int best_label = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      float dist_acc = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(d); ++j) {
        unsigned int data_idx = ((gid * d) + j);
        float data_val = data[data_idx];
        unsigned int cent_idx = ((c * d) + j);
        float cent_val = centroids[cent_idx];
        float diff = (data_val - cent_val);
        float diff_sq = (diff * diff);
        float old_acc = dist_acc;
        dist_acc = (old_acc + diff_sq);
      }
      float curr_dist = dist_acc;
      float curr_best = best_dist;
      bool is_better = (curr_dist < curr_best);
      if (is_better) {
        best_dist = curr_dist;
        best_label = c;
      }
    }
    unsigned int final_label = best_label;
    float final_dist = best_dist;
    labels[gid] = final_label;
    min_distances[gid] = final_dist;
  }
}
