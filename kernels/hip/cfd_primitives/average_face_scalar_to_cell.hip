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
extern "C" __global__ void navatala_cfd_primitives_average_face_scalar_to_cell(const float* facePhi, const float* magSf, const unsigned int* offsets, const unsigned int* faceIdx, const float* sumMag, const int* params, float* outCell) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    unsigned int begU = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    unsigned int endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    float acc = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int fU = faceIdx[k];
      int f = ((int)(fU));
      float term = (magSf[f] * facePhi[f]);
      acc = (acc + term);
    }
    float den = sumMag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (den < __uint_as_float(0x1e3ce508u)) {
      den = __uint_as_float(0x1e3ce508u);
    }
    outCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (acc / den);
  }
}
