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
extern "C" __global__ void navatala_cfd_laplacian_from_face_flux(const float* faceFlux, const int* offsets, const int* faceIdx, const float* signF, const float* vol, const int* counts, float* outLap) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0])) {
    return;
  } else {
    int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + (signF[k] * faceFlux[f]));
    }
    outLap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = s;
  }
}
