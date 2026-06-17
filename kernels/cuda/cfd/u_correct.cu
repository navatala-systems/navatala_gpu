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
extern "C" __global__ void navatala_cfd_u_correct(const float* pCell, const int* owner, const int* neighbour, const float* sfX, const float* sfY, const float* sfZ, const float* bcVal, const int* bcMask, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const float* rAU, const float* ux, const float* uy, const float* uz, const int* counts, const float* paramsF, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float gx = __uint_as_float(0x00000000u);
    float gy = __uint_as_float(0x00000000u);
    float gz = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      int o = owner[f];
      float pf = pCell[o];
      if ((f < counts[2])) {
        int n = neighbour[f];
        pf = (__uint_as_float(0x3f000000u) * (pCell[o] + pCell[n]));
      } else {
        if ((bcMask[f] != 0)) {
          pf = bcVal[f];
        }
      }
      float s = sign[k];
      gx = (gx + (s * (sfX[f] * pf)));
      gy = (gy + (s * (sfY[f] * pf)));
      gz = (gz + (s * (sfZ[f] * pf)));
    }
    float invV = (__uint_as_float(0x3f800000u) / vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    float sC = (paramsF[0] * rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    float dUx = ((gx * invV) * sC);
    float dUy = ((gy * invV) * sC);
    float dUz = ((gz * invV) * sC);
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - dUx);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - dUy);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - dUz);
  }
}
