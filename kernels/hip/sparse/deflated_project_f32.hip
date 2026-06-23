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
extern "C" __global__ void navatala_sparse_deflated_project_f32(const float* x, const float* Z, const float* ZtX, const unsigned int* n, const unsigned int* k, float* w) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float xi = x[gid];
    float proj = __uint_as_float(0x00000000u);
    int kVal = ((int)(k[0]));
    for (int j = 0; j < (int)(kVal); ++j) {
      float zVal = Z[((j * N) + gid)];
      float zx = ZtX[j];
      proj = (proj + (zVal * zx));
    }
    w[gid] = (xi - proj);
  }
}
