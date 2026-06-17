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
extern "C" __global__ void navatala_sparse_strength_symmetric_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const unsigned int* diagIdx, const float* theta, const unsigned int* nRows, unsigned int* strongMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int iDiagPos = ((int)(diagIdx[gid]));
    float diag = values[iDiagPos];
    float absDiagI = abs(diag);
    float thetaVal = theta[0];
    float theta2 = (thetaVal * thetaVal);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float a2 = (a * a);
      int jDiagPos = ((int)(diagIdx[col]));
      float absDiagJ = abs(values[jDiagPos]);
      float threshold = (theta2 * (absDiagI * absDiagJ));
      unsigned int isStrong = ((((col != gid) && (a2 >= threshold))) ? (1u) : (0u));
      strongMask[k] = isStrong;
    }
  }
}
