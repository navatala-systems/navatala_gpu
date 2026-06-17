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
extern "C" __global__ void navatala_cfd_scatter_vec3_and_mask(const int* procFaces, const float* recvVals, const int* counts, float* bcX, float* bcY, float* bcZ, unsigned int* bcMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0])))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 3);
    int b1 = (base + 1);
    int b2 = (base + 2);
    bcX[f] = recvVals[base];
    bcY[f] = recvVals[b1];
    bcZ[f] = recvVals[b2];
    bcMask[f] = ((unsigned int)(1));
  }
}
