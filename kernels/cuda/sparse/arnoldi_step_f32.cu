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
extern "C" __global__ void navatala_sparse_arnoldi_step_f32(const float* Av, const float* V, const float* hCol, const unsigned int* n, const unsigned int* j, float* vNext, float* hNextJ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((gid < N)) {
    float avi = Av[gid];
    float w = avi;
    int jVal = ((int)(j[0]));
    for (int ii = 0; ii < (int)((jVal + 1)); ++ii) {
      float h = hCol[ii];
      float vi = V[((ii * N) + gid)];
      w = (w - (h * vi));
    }
    float wFinal = w;
    vNext[gid] = wFinal;
    atomicAdd(&hNextJ[0], (wFinal * wFinal));
  }
}
