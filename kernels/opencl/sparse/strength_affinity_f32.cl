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

__kernel void navatala_sparse_strength_affinity_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* theta, __global const uint* nRows, __global float* strengthValues, __global uint* strongMask) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float diag = as_float(0x00000000u);
    for (int jd = 0; jd < (int)((re - rs)); ++jd) {
      int kd = (rs + jd);
      int colD = ((int)(colIdx[kd]));
      if ((colD == gid)) {
        diag = values[kd];
      }
    }
    float thetaVal = theta[0];
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float aff = (((col != gid)) ? ((fabs(a) / fabs(diag))) : (as_float(0x00000000u)));
      strengthValues[k] = aff;
      uint isStrong = (((aff >= thetaVal)) ? ((uint)(1u)) : ((uint)(0u)));
      strongMask[k] = isStrong;
    }
  }
}
