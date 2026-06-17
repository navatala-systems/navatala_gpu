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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_backward_f16(const __half* gradC, const __half* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, __half* gradA) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileGradC[256];
  __shared__ float tileBT[256];
  if (valid) {
    float acc = __uint_as_float(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      unsigned int gradCIdx = ((row * nDim) + jIter);
      unsigned int bIdx = ((col * nDim) + jIter);
      __half gradCValF16 = gradC[gradCIdx];
      __half bValF16 = b[bIdx];
      float gradCVal = ((float)(gradCValF16));
      float bVal = ((float)(bValF16));
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    __half finalAccF16 = ((__half)(finalAcc));
    unsigned int gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAccF16;
  }
}
