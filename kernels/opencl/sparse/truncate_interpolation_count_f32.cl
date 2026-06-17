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

__kernel void navatala_sparse_truncate_interpolation_count_f32(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const float* Pvalues, __global const float* threshold, __global const uint* nRows, __global uint* nnzPerRowOut) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float maxAbs = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float v = Pvalues[k];
      maxAbs = (((fabs(v) > maxAbs)) ? (fabs(v)) : (maxAbs));
    }
    float thresh = (maxAbs * threshold[0]);
    uint nnz = (uint)(0u);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float v2 = Pvalues[k2];
      if ((fabs(v2) >= thresh)) {
        nnz = (nnz + (uint)(1u));
      }
    }
    nnzPerRowOut[row] = nnz;
  }
}
