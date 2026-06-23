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
extern "C" __global__ void navatala_cfd_dic_build_reciprocal_d(const float* diag, const float* upper, const int* owner, const int* neighbour, const int* counts, float* rD) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    rD[c] = diag[c];
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float u = upper[f];
    float rdo = rD[o];
    float rdn = rD[n];
    rD[n] = (rdn - ((u * u) / rdo));
  }
  for (int c2 = 0; c2 < (int)(counts[0]); ++c2) {
    float v = rD[c2];
    if ((v != __uint_as_float(0x00000000u))) {
      rD[c2] = (__uint_as_float(0x3f800000u) / v);
    } else {
      rD[c2] = __uint_as_float(0x00000000u);
    }
  }
}
