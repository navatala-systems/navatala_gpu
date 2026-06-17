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
extern "C" __global__ void navatala_cfd_chebyshev_smooth_post_fused_mixed(const float* Az, const float* r, const float* diagInv, const float* dPrev, const double* coeffA, const double* coeffB, const unsigned int* n, float* z, float* dNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    double cA = coeffA[0];
    double cB = coeffB[0];
    double az_i = ((double)(Az[i]));
    double r_i = ((double)(r[i]));
    double di_i = ((double)(diagInv[i]));
    double dp_i = ((double)(dPrev[i]));
    double z_i = ((double)(z[i]));
    double res = (r_i - az_i);
    double dinvr = (di_i * res);
    double corr = ((cA * dinvr) + (cB * dp_i));
    z[i] = ((float)((z_i + corr)));
    dNew[i] = ((float)(corr));
  }
}
