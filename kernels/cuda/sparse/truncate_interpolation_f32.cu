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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_truncate_interpolation_f32(const unsigned int* ProwPtr, const unsigned int* PcolIdx, const float* Pvalues, const float* threshold, const unsigned int* nRows, const unsigned int* ProwPtrOut, unsigned int* PcolIdxOut, float* PvaluesOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    int pStart = ((int)(ProwPtrOut[row]));
    float maxAbs = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float v = Pvalues[k];
      maxAbs = (((abs(v) > maxAbs)) ? (abs(v)) : (maxAbs));
    }
    float thresh = (maxAbs * threshold[0]);
    int pos = 0;
    for (int j3 = 0; j3 < (int)((re - rs)); ++j3) {
      int k3 = (rs + j3);
      float v3 = Pvalues[k3];
      if ((abs(v3) >= thresh)) {
        PcolIdxOut[(pStart + pos)] = PcolIdx[k3];
        PvaluesOut[(pStart + pos)] = v3;
        pos = (pos + 1);
      }
    }
  }
}
