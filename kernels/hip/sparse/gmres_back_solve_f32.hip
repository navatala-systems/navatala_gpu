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
extern "C" __global__ void navatala_sparse_gmres_back_solve_f32(const float* H, const float* g, const unsigned int* m, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int M = ((int)(m[0]));
  int i = (M - 1);
  float gi = g[i];
  float hii = H[((i * M) + i)];
  y[i] = (gi / hii);
}
