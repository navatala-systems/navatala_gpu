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
extern "C" __global__ void navatala_cfd_pressure_face_flux_correction_corrected(const int* owner, const int* neighbour, const float* weights, const float* gradX, const float* gradY, const float* gradZ, const float* magSf, const float* dCorrX, const float* dCorrY, const float* dCorrZ, const float* rAUf, const unsigned int* params, float* outFaceFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[1]))) {
      outFaceFlux[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
    } else {
      int own = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      int nei = neighbour[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float w = weights[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float omw = (__uint_as_float(0x3f800000u) - w);
      float gx = ((w * gradX[own]) + (omw * gradX[nei]));
      float gy = ((w * gradY[own]) + (omw * gradY[nei]));
      float gz = ((w * gradZ[own]) + (omw * gradZ[nei]));
      float corr = (((dCorrX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * gx) + (dCorrY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * gy)) + (dCorrZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * gz));
      outFaceFlux[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (rAUf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * (magSf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * corr));
    }
  }
}
