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
extern "C" __global__ void navatala_cfd_mg_coarse_rhs(const float* b, const float* ax, float* r, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    r[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (b[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
  }
}
