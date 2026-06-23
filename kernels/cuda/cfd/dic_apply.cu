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
extern "C" __global__ void navatala_cfd_dic_apply(const float* r, const float* rD, const float* upper, const int* owner, const int* neighbour, const int* counts, float* z) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    z[c] = (rD[c] * r[c]);
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float zn = z[n];
    z[n] = (zn - ((rD[n] * upper[f]) * z[o]));
  }
  for (int t = 0; t < (int)(counts[2]); ++t) {
    int f2 = ((counts[2] - 1) - t);
    int o2 = owner[f2];
    int n2 = neighbour[f2];
    float zo = z[o2];
    z[o2] = (zo - ((rD[o2] * upper[f2]) * z[n2]));
  }
}
