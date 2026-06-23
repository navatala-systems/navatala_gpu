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
extern "C" __global__ void navatala_cfd_mg_strength_of_connection(const float* diag, const float* upper, const int* owner, const int* nei, float* strength, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[1]))) {
    return;
  } else {
    int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int n = nei[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float dO = abs(diag[o]);
    float dN = abs(diag[n]);
    float denomSq = (dO * dN);
    if (denomSq > __uint_as_float(0x0da24260u)) {
      strength[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (abs(upper[(int)(blockIdx.x * blockDim.x + threadIdx.x)]) / sqrt(denomSq));
    } else {
      strength[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
    }
  }
}
