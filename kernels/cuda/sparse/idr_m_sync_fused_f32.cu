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
extern "C" __global__ void navatala_sparse_idr_m_sync_fused_f32(const float* P, const float* r, const float* Minv, const unsigned int* n, const unsigned int* s, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int k = (int)(blockIdx.x);
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  float acc = __uint_as_float(0x00000000u);
  for (int j = 0; j < (int)(S); ++j) {
    float pjr = (((gid < N)) ? ((P[(gid + (j * N))] * r[gid])) : (__uint_as_float(0x00000000u)));
    float mij = Minv[(k + (j * S))];
    acc = (acc + (mij * pjr));
  }
  float warpSum = gpu_warp_reduce_sum(acc);
  __shared__ float sdata[32];
  int lane = (int)(threadIdx.x % warpSize);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  __syncthreads();
  if (lid < 8) {
    float sv = sdata[lid];
    float finalSum = gpu_warp_reduce_sum(sv);
    if (lid == 0) {
      c[k] = finalSum;
    }
  }
}
