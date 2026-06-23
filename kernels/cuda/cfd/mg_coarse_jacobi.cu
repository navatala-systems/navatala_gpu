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
extern "C" __global__ void navatala_cfd_mg_coarse_jacobi(float* x, const float* r, const float* diag, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    float d = diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (d == __uint_as_float(0x00000000u)) {
      return;
    } else {
      float omega = __uint_as_float(0x3f4ccccdu);
      x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + (omega * (r[(int)(blockIdx.x * blockDim.x + threadIdx.x)] / d)));
    }
  }
}
