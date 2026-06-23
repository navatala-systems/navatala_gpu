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
extern "C" __global__ void navatala_cfd_scalar_ldu_mat_vec(const float* diag, const float* upper, const float* lower, const float* x, const int* owner, const int* neighbour, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const unsigned int* counts, float* ax) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0])))) {
    return;
  } else {
    float s = (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * x[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
    int beg = ((int)(offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int f = faceIdx[k];
      int sg = sign[k];
      if ((((int)(f)) < ((int)(counts[1])))) {
        if ((sg >= 0)) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = s;
  }
}
