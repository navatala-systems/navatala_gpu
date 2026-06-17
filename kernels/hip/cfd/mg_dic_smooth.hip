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
extern "C" __global__ void navatala_cfd_mg_dic_smooth(const float* b, const float* ax, const double* rD, const int* edgeU, const int* edgeV, const float* edgeCf, const int* counts, float* x, float* z) {
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
      z[c] = ((float)((rD[c] * (((double)(b[c])) - ((double)(ax[c]))))));
    }
    for (int e = 0; e < (int)(nEdges); ++e) {
      int v = edgeV[e];
      int u = edgeU[e];
      z[v] = ((float)((((double)(z[v])) - (rD[v] * (((double)(edgeCf[e])) * ((double)(z[u])))))));
    }
    for (int t = 0; t < (int)(nEdges); ++t) {
      int e = ((nEdges - 1) - t);
      int u = edgeU[e];
      int v = edgeV[e];
      z[u] = ((float)((((double)(z[u])) - (rD[u] * (((double)(edgeCf[e])) * ((double)(z[v])))))));
    }
    for (int c2 = 0; c2 < (int)(nCells); ++c2) {
      x[c2] = (x[c2] + z[c2]);
    }
  }
}
