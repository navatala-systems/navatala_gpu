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

__kernel void navatala_sparse_bsr_sp_m_v_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* x, __global const uint* nBlockRows, __global const uint* blockDim, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int blockRow = (int)(get_global_id(0));
  int N = ((int)(nBlockRows[0]));
  int bd = ((int)(blockDim[0]));
  if ((blockRow < N)) {
    int rs = ((int)(rowPtr[blockRow]));
    int re = ((int)(rowPtr[(blockRow + 1)]));
    for (int comp = 0; comp < (int)(bd); ++comp) {
      float sum = as_float(0x00000000u);
      for (int jj = 0; jj < (int)((re - rs)); ++jj) {
        int blk = (rs + jj);
        int blkCol = ((int)(colIdx[blk]));
        for (int c2 = 0; c2 < (int)(bd); ++c2) {
          float a = values[(((((blk * bd) * bd) + (comp * bd)) + c2) + 0)];
          float xval = x[((blkCol * bd) + c2)];
          sum = (sum + (a * xval));
        }
      }
      y[((blockRow * bd) + comp)] = sum;
    }
  }
}
