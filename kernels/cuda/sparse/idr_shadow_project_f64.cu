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
extern "C" __global__ void navatala_sparse_idr_shadow_project_f64(const double* P, const double* r, const unsigned int* n, const unsigned int* s, double* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int k = (int)(blockIdx.x);
  int N = ((int)(n[0]));
  double val = (((gid < N)) ? ((P[(gid + (k * N))] * r[gid])) : (__longlong_as_double(0x0000000000000000ull)));
  double warpSum = gpu_warp_reduce_sum(val);
  __shared__ double sdata[32];
  int lane = (int)(threadIdx.x % warpSize);
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  __syncthreads();
  if ((lid < 8)) {
    double sv = sdata[lid];
    double finalSum = gpu_warp_reduce_sum(sv);
    if ((lid == 0)) {
      c[k] = finalSum;
    }
  }
}
