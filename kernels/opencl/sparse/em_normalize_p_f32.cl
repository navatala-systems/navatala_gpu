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

__kernel void navatala_sparse_em_normalize_p_f32(__global const uint* ProwPtr, __global const uint* nRows, __global float* Pvalues) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(ProwPtr[gid]));
    int re = ((int)(ProwPtr[(gid + 1)]));
    float rowSum = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      rowSum = (rowSum + Pvalues[k]);
    }
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float v = Pvalues[k2];
      Pvalues[k2] = (v / rowSum);
    }
  }
}
