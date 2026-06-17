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

__kernel void navatala_sparse_build_smoothed_prolongation_f32(__global const uint* ArowPtr, __global const uint* AcolIdx, __global const float* Avalues, __global const float* diagInv, __global const uint* ProwPtr, __global const uint* PcolIdx, __global const float* Pvalues, __global const float* omega, __global const uint* nRows, __global float* PsmoothValues) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    uint myAgg = PcolIdx[row];
    float myPval = Pvalues[row];
    int aStart = ((int)(ArowPtr[row]));
    int aEnd = ((int)(ArowPtr[(row + 1)]));
    float sumAP = as_float(0x00000000u);
    for (int k = 0; k < (int)((aEnd - aStart)); ++k) {
      int idx = (aStart + k);
      int col = ((int)(AcolIdx[idx]));
      float aVal = Avalues[idx];
      uint colAgg = PcolIdx[col];
      if ((colAgg == myAgg)) {
        sumAP = (sumAP + aVal);
      }
    }
    float om = omega[0];
    float di = diagInv[row];
    float correction = (om * (di * sumAP));
    PsmoothValues[row] = (myPval - correction);
  }
}
