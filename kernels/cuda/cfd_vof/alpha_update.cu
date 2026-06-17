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
extern "C" __global__ void navatala_cfd_vof_alpha_update(const float* alpha, const float* divA, const float* paramsF, const int* counts, const float* rSubDeltaT, float* alphaNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    float a = (alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (divA[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] / rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    if ((a < __uint_as_float(0x00000000u))) {
      a = __uint_as_float(0x00000000u);
    }
    if ((a > __uint_as_float(0x3f800000u))) {
      a = __uint_as_float(0x3f800000u);
    }
    alphaNew[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = a;
  }
}
