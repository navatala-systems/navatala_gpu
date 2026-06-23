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
extern "C" __global__ void navatala_cfd_mg_coarse_matvec(const int* edgeU, const int* edgeV, const float* edgeCf, const float* diag, const float* x, float* outAx, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = (((int)(mgCounts[1])) + ((int)(mgCounts[2])));
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= total) {
    return;
  } else {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) < ((int)(mgCounts[1]))) {
      atomicAdd(&outAx[(int)(blockIdx.x * blockDim.x + threadIdx.x)], (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * x[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
    } else {
      int e = ((int)(blockIdx.x * blockDim.x + threadIdx.x) - ((int)(mgCounts[1])));
      int u = edgeU[e];
      int v = edgeV[e];
      float cf = edgeCf[e];
      atomicAdd(&outAx[u], (cf * x[v]));
      atomicAdd(&outAx[v], (cf * x[u]));
    }
  }
}
