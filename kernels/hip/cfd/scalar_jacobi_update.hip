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
extern "C" __global__ void navatala_cfd_scalar_jacobi_update(const float* ax, const float* diag, const float* rhs, const float* params, const unsigned int* counts, float* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0])))) {
    return;
  } else {
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float absD = abs(d);
    if ((absD < smallDiag)) {
      if ((d >= __uint_as_float(0x00000000u))) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float xi = x[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float correction = (omega * ((rhs[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)]) / d));
    x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (xi + correction);
  }
}
