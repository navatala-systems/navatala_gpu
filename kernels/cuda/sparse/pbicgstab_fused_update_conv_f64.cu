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
extern "C" __global__ void navatala_sparse_pbicgstab_fused_update_conv_f64(const double* phat, const double* shat, const double* s, const double* t, const double* alpha, const double* omega, const unsigned int* n, double* x, double* r, double* rrPartials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  if (inBounds) {
    double a = alpha[0];
    double om = omega[0];
    double ph = phat[gid];
    double sh = shat[gid];
    double si = s[gid];
    double ti = t[gid];
    double xi = x[gid];
    double ri = (si - (om * ti));
    x[gid] = (xi + ((a * ph) + (om * sh)));
    r[gid] = ri;
  }
  double sq = ((inBounds) ? ((r[gid] * r[gid])) : (__longlong_as_double(0x0000000000000000ull)));
  double warpSum = gpu_warp_reduce_sum(sq);
  __shared__ double sdata[32];
  int lane = (int)(threadIdx.x % warpSize);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  __syncthreads();
  if (lid < 8) {
    double val = sdata[lid];
    double finalSum = gpu_warp_reduce_sum(val);
    if (lid == 0) {
      int grpId = (int)(blockIdx.x);
      rrPartials[grpId] = finalSum;
    }
  }
}
