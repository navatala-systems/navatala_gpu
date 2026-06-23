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

__kernel void navatala_sparse_truncate_interpolation_f32(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const float* Pvalues, __global const float* threshold, __global const uint* nRows, __global const uint* ProwPtrOut, __global uint* PcolIdxOut, __global float* PvaluesOut) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    int pStart = ((int)(ProwPtrOut[row]));
    float maxAbs = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float v = Pvalues[k];
      maxAbs = (((fabs(v) > maxAbs)) ? (fabs(v)) : (maxAbs));
    }
    float thresh = (maxAbs * threshold[0]);
    int pos = 0;
    for (int j3 = 0; j3 < (int)((re - rs)); ++j3) {
      int k3 = (rs + j3);
      float v3 = Pvalues[k3];
      if (fabs(v3) >= thresh) {
        PcolIdxOut[(pStart + pos)] = PcolIdx[k3];
        PvaluesOut[(pStart + pos)] = v3;
        pos = (pos + 1);
      }
    }
  }
}
