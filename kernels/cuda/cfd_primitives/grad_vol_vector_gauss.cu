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
extern "C" __global__ void navatala_cfd_primitives_grad_vol_vector_gauss(const float* cellUx, const float* cellUy, const float* cellUz, const float* bcX, const float* bcY, const float* bcZ, const unsigned int* bcMask, const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* params, float* outXX, float* outXY, float* outXZ, float* outYX, float* outYY, float* outYZ, float* outZX, float* outZY, float* outZZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    float sumXX = __uint_as_float(0x00000000u);
    float sumXY = __uint_as_float(0x00000000u);
    float sumXZ = __uint_as_float(0x00000000u);
    float sumYX = __uint_as_float(0x00000000u);
    float sumYY = __uint_as_float(0x00000000u);
    float sumYZ = __uint_as_float(0x00000000u);
    float sumZX = __uint_as_float(0x00000000u);
    float sumZY = __uint_as_float(0x00000000u);
    float sumZZ = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        float s = sign[k];
        int o = owner[f];
        float uxO = cellUx[o];
        float uyO = cellUy[o];
        float uzO = cellUz[o];
        float uxF = uxO;
        float uyF = uyO;
        float uzF = uzO;
        if ((f < params[2])) {
          int n = neighbour[f];
          float uxN = cellUx[n];
          float uyN = cellUy[n];
          float uzN = cellUz[n];
          float ww = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - ww);
          uxF = ((ww * uxO) + (iw * uxN));
          uyF = ((ww * uyO) + (iw * uyN));
          uzF = ((ww * uzO) + (iw * uzN));
        } else {
          unsigned int m = bcMask[f];
          if ((m == 1u)) {
            float ww = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - ww);
            uxF = ((ww * uxO) + (iw * bcX[f]));
            uyF = ((ww * uyO) + (iw * bcY[f]));
            uzF = ((ww * uzO) + (iw * bcZ[f]));
          }
          if ((m == 2u)) {
            uxF = bcX[f];
            uyF = bcY[f];
            uzF = bcZ[f];
          }
        }
        float sfXf = sfX[f];
        float sfYf = sfY[f];
        float sfZf = sfZ[f];
        float wUx = (s * uxF);
        float wUy = (s * uyF);
        float wUz = (s * uzF);
        sumXX = (sumXX + (wUx * sfXf));
        sumXY = (sumXY + (wUx * sfYf));
        sumXZ = (sumXZ + (wUx * sfZf));
        sumYX = (sumYX + (wUy * sfXf));
        sumYY = (sumYY + (wUy * sfYf));
        sumYZ = (sumYZ + (wUy * sfZf));
        sumZX = (sumZX + (wUz * sfXf));
        sumZY = (sumZY + (wUz * sfYf));
        sumZZ = (sumZZ + (wUz * sfZf));
      }
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if ((v != __uint_as_float(0x00000000u))) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outXX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumXX * invV);
    outXY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumXY * invV);
    outXZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumXZ * invV);
    outYX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumYX * invV);
    outYY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumYY * invV);
    outYZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumYZ * invV);
    outZX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZX * invV);
    outZY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZY * invV);
    outZZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZZ * invV);
  }
}
