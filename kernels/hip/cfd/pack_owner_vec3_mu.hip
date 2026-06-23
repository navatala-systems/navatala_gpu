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
extern "C" __global__ void navatala_cfd_pack_owner_vec3_mu(const float* ux, const float* uy, const float* uz, const float* muCell, const int* owner, const int* procFaces, const int* counts, float* outSend) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0])))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int o = owner[f];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 4);
    outSend[base] = ux[o];
    int b1 = (base + 1);
    outSend[b1] = uy[o];
    int b2 = (base + 2);
    outSend[b2] = uz[o];
    int b3 = (base + 3);
    outSend[b3] = muCell[o];
  }
}
