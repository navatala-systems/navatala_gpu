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
extern "C" __global__ void navatala_graph_spmv_weighted_f32(const unsigned int* offsets, const unsigned int* indices, const float* weights, const float* x, const unsigned int* numVertices, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int base = offsets[gid];
    unsigned int endv = offsets[(gid + 1u)];
    unsigned int rowlen = (endv - base);
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      unsigned int eidx = (base + ((unsigned int)(k)));
      unsigned int col = indices[eidx];
      float w = weights[eidx];
      float xv = x[col];
      acc = (acc + (w * xv));
    }
    y[gid] = acc;
  }
}
