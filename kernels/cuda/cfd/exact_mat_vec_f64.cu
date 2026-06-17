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
extern "C" __global__ void navatala_cfd_exact_mat_vec_f64(const float* x, const int* owner, const int* neighbour, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const double* upper, const double* lower, const double* diag, const int* counts, float* outAx) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0])))) {
    return;
  } else {
    double s = (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ((double)(x[(int)(blockIdx.x * blockDim.x + threadIdx.x)])));
    unsigned int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    unsigned int end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      unsigned int f = faceIdx[k];
      if ((((int)(f)) < ((int)(counts[2])))) {
        int sg = sign[k];
        if ((sg >= 0)) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * ((double)(x[nb]))));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * ((double)(x[ow]))));
        }
      }
    }
    outAx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ((float)(s));
  }
}
