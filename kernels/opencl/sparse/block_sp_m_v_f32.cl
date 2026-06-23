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

__kernel void navatala_sparse_block_sp_m_v_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* X, __global const uint* nRows, __global const uint* k, __global float* Y) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int total = (((int)(nRows[0])) * ((int)(k[0])));
  int nR = ((int)(nRows[0]));
  if (gid < total) {
    int row = (gid % nR);
    int vecIdx = (gid / nR);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k2 = (rs + j);
      int col = ((int)(colIdx[k2]));
      float a = values[k2];
      float x = X[((vecIdx * nR) + col)];
      sum = (sum + (a * x));
    }
    Y[gid] = sum;
  }
}
