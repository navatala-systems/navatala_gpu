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
extern "C" __global__ void navatala_cfd_face_flux(const float* x, const int* owner, const int* neighbour, const float* cf, const float* bcVal, const int* bcMask, const int* counts, float* outFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float xo = x[o];
    float other = xo;
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2])) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      other = x[n];
    } else {
      int m = bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (((m == 1) || (m == 3))) {
        other = bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      } else {
        if ((m == 2)) {
          other = __uint_as_float(0x00000000u);
        } else {
          if ((m == 4)) {
            outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
            return;
          }
        }
      }
    }
    outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (other - xo));
  }
}
