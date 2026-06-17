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

__kernel void navatala_linalg_reduce_col_sum_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint col = gid;
  if ((col < n[(uint)(0u)])) {
    float sumAccum = as_float(0x00000000u);
    for (int i = 0; i < (int)(m[(uint)(0u)]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[(uint)(0u)]) + col);
      float val = A[idx];
      float currentSum = sumAccum;
      float nextSum = (currentSum + val);
      sumAccum = nextSum;
    }
    float finalSum = sumAccum;
    result[col] = finalSum;
  }
}
