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

__kernel void navatala_sparse_mg_weighted_restrict_f32(__global const uint* RrowPtr, __global const uint* RcolIdx, __global const float* Rvalues, __global const float* rFine, __global const float* weights, __global const uint* nCoarse, __global float* rCoarse) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nCoarse[0]));
  if (row < N) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      float rv = Rvalues[k];
      float rf = rFine[col];
      float w = weights[col];
      acc = (acc + (rv * (w * rf)));
    }
    rCoarse[row] = acc;
  }
}
