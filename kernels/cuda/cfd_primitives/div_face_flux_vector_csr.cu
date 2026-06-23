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
extern "C" __global__ void navatala_cfd_primitives_div_face_flux_vector_csr(const int* offsets, const int* faceIdx, const float* sign, const float* faceFluxX, const float* faceFluxY, const float* faceFluxZ, const float* vol, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float accX = __uint_as_float(0x00000000u);
    float accY = __uint_as_float(0x00000000u);
    float accZ = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      accX = (accX + (s * faceFluxX[f]));
      accY = (accY + (s * faceFluxY[f]));
      accZ = (accZ + (s * faceFluxZ[f]));
    }
    float volCell = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float volSafe = (((abs(volCell) > __uint_as_float(0x1e3ce508u))) ? (volCell) : (__uint_as_float(0x1e3ce508u)));
    float invV = (__uint_as_float(0x3f800000u) / volSafe);
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (accX * invV);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (accY * invV);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (accZ * invV);
  }
}
