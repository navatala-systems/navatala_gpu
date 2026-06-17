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
extern "C" __global__ void navatala_cfd_vof_phir(const float* gx, const float* gy, const float* gz, const float* sfX, const float* sfY, const float* sfZ, const float* phi, const float* magSf, const int* owner, const int* nei, const int* faceParams, const float* paramsF, float* outPhir) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= faceParams[0])) {
    return;
  } else {
    if ((paramsF[0] == __uint_as_float(0x00000000u))) {
      outPhir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      return;
    }
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gox = gx[o];
    float goy = gy[o];
    float goz = gz[o];
    float gnx = gx[n];
    float gny = gy[n];
    float gnz = gz[n];
    float gfx = (__uint_as_float(0x3f000000u) * (gox + gnx));
    float gfy = (__uint_as_float(0x3f000000u) * (goy + gny));
    float gfz = (__uint_as_float(0x3f000000u) * (goz + gnz));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float dotSf = (((gfx * sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) + (gfy * sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) + (gfz * sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    float nHatf = (dotSf / (magg + paramsF[1]));
    float phic = ((paramsF[0] * abs(phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) / magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outPhir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phic * nHatf);
  }
}
