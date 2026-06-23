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
extern "C" __global__ void navatala_cfd_grad_alpha_cell_f64(const double* alphaF, const double* sfX, const double* sfY, const double* sfZ, const int* offsets, const int* faceIdx, const float* sign, const double* vol, const int* counts, double* gx, double* gy, double* gz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double sx = __longlong_as_double(0x0000000000000000ull);
    double sy = __longlong_as_double(0x0000000000000000ull);
    double sz = __longlong_as_double(0x0000000000000000ull);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double aV = ((double)(alphaF[f]));
      double s = (signV * aV);
      double sfXv = ((double)(sfX[f]));
      double sfYv = ((double)(sfY[f]));
      double sfZv = ((double)(sfZ[f]));
      sx = (sx + (s * sfXv));
      sy = (sy + (s * sfYv));
      sz = (sz + (s * sfZv));
    }
    double invV = (__longlong_as_double(0x3ff0000000000000ull) / ((double)(vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])));
    gx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sx * invV);
    gy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sy * invV);
    gz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sz * invV);
  }
}
