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
extern "C" __global__ void navatala_cfd_two_mag_sqr_symm_grad_u(const float* ufX, const float* ufY, const float* ufZ, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* counts, float* outS2) {
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
    float g00 = __uint_as_float(0x00000000u);
    float g01 = __uint_as_float(0x00000000u);
    float g02 = __uint_as_float(0x00000000u);
    float g10 = __uint_as_float(0x00000000u);
    float g11 = __uint_as_float(0x00000000u);
    float g12 = __uint_as_float(0x00000000u);
    float g20 = __uint_as_float(0x00000000u);
    float g21 = __uint_as_float(0x00000000u);
    float g22 = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float sx = (s * sfX[f]);
      float sy = (s * sfY[f]);
      float sz = (s * sfZ[f]);
      float ux = ufX[f];
      float uy = ufY[f];
      float uz = ufZ[f];
      g00 = (g00 + (sx * ux));
      g01 = (g01 + (sx * uy));
      g02 = (g02 + (sx * uz));
      g10 = (g10 + (sy * ux));
      g11 = (g11 + (sy * uy));
      g12 = (g12 + (sy * uz));
      g20 = (g20 + (sz * ux));
      g21 = (g21 + (sz * uy));
      g22 = (g22 + (sz * uz));
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if ((v != __uint_as_float(0x00000000u))) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    g00 = (g00 * invV);
    g01 = (g01 * invV);
    g02 = (g02 * invV);
    g10 = (g10 * invV);
    g11 = (g11 * invV);
    g12 = (g12 * invV);
    g20 = (g20 * invV);
    g21 = (g21 * invV);
    g22 = (g22 * invV);
    float s00 = g00;
    float s11 = g11;
    float s22 = g22;
    float s01 = (__uint_as_float(0x3f000000u) * (g01 + g10));
    float s02 = (__uint_as_float(0x3f000000u) * (g02 + g20));
    float s12 = (__uint_as_float(0x3f000000u) * (g12 + g21));
    float d0 = (s00 * s00);
    float d1 = (s11 * s11);
    float d2 = (s22 * s22);
    float o0 = (s01 * s01);
    float o1 = (s02 * s02);
    float o2 = (s12 * s12);
    float diag = ((d0 + d1) + d2);
    float off = ((o0 + o1) + o2);
    outS2[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((__uint_as_float(0x40000000u) * diag) + (__uint_as_float(0x40800000u) * off));
  }
}
