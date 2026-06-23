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
extern "C" __global__ void navatala_graph_compute_degree_matrix_f32(const float* W, const unsigned int* n, float* D) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      float wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int degF32RedStride = 128u;
    for (int degF32RedStep = 0; degF32RedStep < (int)(8); ++degF32RedStep) {
      unsigned int degF32Stride = degF32RedStride;
      if (lid < degF32Stride) {
        float other = sdata[(lid + degF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int degF32StrideToHalve = degF32RedStride;
      unsigned int degF32NextStride = (degF32StrideToHalve >> 1u);
      degF32RedStride = degF32NextStride;
      __syncthreads();
    }
    if (lid == 0u) {
      float degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}
