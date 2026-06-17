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
extern "C" __global__ void navatala_sparse_gmres_update_solution_f32(const float* V, const float* y, const unsigned int* n, const unsigned int* m, float* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  int M = ((int)(m[0]));
  if ((gid < N)) {
    float acc = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(M); ++j) {
      float yj = y[j];
      float vij = V[(gid + (j * N))];
      acc = (acc + (yj * vij));
    }
    float xi = x[gid];
    x[gid] = (xi + acc);
  }
}
