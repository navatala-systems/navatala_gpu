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
extern "C" __global__ void navatala_cfd_k_omega_s_s_t_nut_compute(const float* kVals, const float* omegaVals, const float* f23Vals, const float* s2Vals, const unsigned int* counts, const float* params, float* outNut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float a1 = params[0];
    float b1 = params[1];
    float k = kVals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float om = omegaVals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float f23 = f23Vals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float s2raw = s2Vals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float denomA = (a1 * om);
    float denomB = ((b1 * f23) * sqrt((s2raw * ((float)((s2raw > __uint_as_float(0x00000000u)))))));
    float num = (a1 * k);
    if (denomA > denomB) {
      if (denomA > __uint_as_float(0x00000000u)) {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (num / denomA);
      } else {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
      }
    } else {
      if (denomB > __uint_as_float(0x00000000u)) {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (num / denomB);
      } else {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
      }
    }
  }
}
