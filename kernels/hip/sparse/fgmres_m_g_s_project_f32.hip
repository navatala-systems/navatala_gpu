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
extern "C" __global__ void navatala_sparse_fgmres_m_g_s_project_f32(const float* vi, const unsigned int* n, float* w, float* hij) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  float prod = ((inBounds) ? ((w[gid] * vi[gid])) : (__uint_as_float(0x00000000u)));
  float warpSum = gpu_warp_reduce_sum(prod);
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
      hij[0] = finalSum;
    }
  }
}
