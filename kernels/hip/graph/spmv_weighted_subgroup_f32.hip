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
template <typename T>
__device__ inline T gpu_warp_reduce_sum(T v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}
extern "C" __global__ void navatala_graph_spmv_weighted_subgroup_f32(const unsigned int* offsets, const unsigned int* indices, const float* weights, const float* x, const unsigned int* numVertices, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lane = ((unsigned int)((int)(threadIdx.x % warpSize)));
  unsigned int subgroupSize = ((unsigned int)((int)(warpSize)));
  unsigned int row = (gid / subgroupSize);
  unsigned int numV = numVertices[0];
  if ((row < numV)) {
    unsigned int base = offsets[row];
    unsigned int endv = offsets[(row + 1u)];
    unsigned int rowlen = (endv - base);
    unsigned int rowIters = ((rowlen + (subgroupSize - 1u)) / subgroupSize);
    float laneAcc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      unsigned int rel = ((((unsigned int)(k)) * subgroupSize) + lane);
      if ((rel < rowlen)) {
        unsigned int eidx = (base + rel);
        unsigned int col = indices[eidx];
        float w = weights[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + (w * xv));
      }
    }
    float rowSum = gpu_warp_reduce_sum(laneAcc);
    if ((lane == 0u)) {
      y[row] = rowSum;
    }
  }
}
