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
extern "C" __global__ void navatala_cfd_div_upwind_coeffs(const float* weights, const float* flux, const int* counts, float* outLower, float* outUpper) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float phi = flux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float low = ((__uint_as_float(0x00000000u) - w) * phi);
    float up = (low + phi);
    outLower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = low;
    outUpper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = up;
  }
}
