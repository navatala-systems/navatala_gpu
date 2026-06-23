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
template <typename T>
__device__ inline T gpu_warp_reduce_sum(T v) {
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}
extern "C" __global__ void navatala_sparse_mg_error_norm_partials_f32(const float* error, const unsigned int* n, float* partials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((error[gid] * error[gid])) : (__uint_as_float(0x00000000u)));
  float warpSum = gpu_warp_reduce_sum(sq);
  __shared__ float sdata[32];
  int lane = (int)(threadIdx.x % warpSize);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  __syncthreads();
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = gpu_warp_reduce_sum(val);
    if (lid == 0) {
      int grpId = (int)(blockIdx.x);
      partials[grpId] = finalSum;
    }
  }
}
