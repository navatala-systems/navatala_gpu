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
extern "C" __global__ void navatala_cfd_mg_dic_build(const float* diag, const int* edgeU, const int* edgeV, const float* edgeCf, const int* counts, double* rD) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1)) {
    return;
  } else {
    int nCells = counts[1];
    int nEdges = counts[2];
    for (int c = 0; c < (int)(nCells); ++c) {
      rD[c] = ((double)(diag[c]));
    }
    for (int e = 0; e < (int)(nEdges); ++e) {
      int u = edgeU[e];
      int v = edgeV[e];
      double dl = rD[u];
      if ((dl > __longlong_as_double(0x01a56e1fc2f8f359ull))) {
        double cf = ((double)(edgeCf[e]));
        rD[v] = (rD[v] - ((cf * cf) / dl));
      }
    }
    for (int c2 = 0; c2 < (int)(nCells); ++c2) {
      double d = rD[c2];
      if ((d > __longlong_as_double(0x01a56e1fc2f8f359ull))) {
        rD[c2] = (__longlong_as_double(0x3ff0000000000000ull) / d);
      } else {
        rD[c2] = __longlong_as_double(0x0000000000000000ull);
      }
    }
  }
}
