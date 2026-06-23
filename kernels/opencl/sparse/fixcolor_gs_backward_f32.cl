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

__kernel void navatala_sparse_fixcolor_gs_backward_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* diag, __global const float* b, __global const uint* reorderPerm, __global const uint* colorOffsets, __global const uint* nRowsThisColor, __global float* x) {
  int gid0 = (int)get_global_id(0);
  int localIdx = (int)(get_global_id(0));
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int start = ((int)(colorOffsets[0]));
    int permIdx = (start + localIdx);
    int row = ((int)(reorderPerm[permIdx]));
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float bi = b[row];
    float di = diag[row];
    x[row] = ((bi - sum) / di);
  }
}
