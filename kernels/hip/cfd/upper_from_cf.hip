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
extern "C" __global__ void navatala_cfd_upper_from_cf(const float* cf, const int* counts, float* upper) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[2])) > 0 ? ((int)(counts[2])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[2]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]) {
    return;
  } else {
    upper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
  }
}
