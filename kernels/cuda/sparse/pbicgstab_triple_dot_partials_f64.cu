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
extern "C" __global__ void navatala_sparse_pbicgstab_triple_dot_partials_f64(const double* r0hat, const double* r, const double* v, const double* s, const unsigned int* n, double* partials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  double h0 = ((inBounds) ? (r0hat[gid]) : (__longlong_as_double(0x0000000000000000ull)));
  double d0 = ((inBounds) ? ((h0 * r[gid])) : (__longlong_as_double(0x0000000000000000ull)));
  double d1 = ((inBounds) ? ((h0 * v[gid])) : (__longlong_as_double(0x0000000000000000ull)));
  double d2 = ((inBounds) ? ((h0 * s[gid])) : (__longlong_as_double(0x0000000000000000ull)));
  double ws0 = gpu_warp_reduce_sum(d0);
  double ws1 = gpu_warp_reduce_sum(d1);
  double ws2 = gpu_warp_reduce_sum(d2);
  __shared__ double sd0[32];
  __shared__ double sd1[32];
  __shared__ double sd2[32];
  int lane = (int)(threadIdx.x % warpSize);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sd0[warpIdx] = ws0;
    sd1[warpIdx] = ws1;
    sd2[warpIdx] = ws2;
  }
  __syncthreads();
  if (lid < 8) {
    int grpId = (int)(blockIdx.x);
    int base = (grpId * 3);
    double f0 = gpu_warp_reduce_sum(sd0[lid]);
    double f1 = gpu_warp_reduce_sum(sd1[lid]);
    double f2 = gpu_warp_reduce_sum(sd2[lid]);
    if (lid == 0) {
      partials[base] = f0;
      partials[(base + 1)] = f1;
      partials[(base + 2)] = f2;
    }
  }
}
