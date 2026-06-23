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
extern "C" __global__ void navatala_sparse_bicgstab_fused_update_all_f32(const float* phat, const float* shat, const float* s, const float* t, const float* alpha, const float* omega, const unsigned int* n, float* x, float* r) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float om = omega[0];
    float ph = phat[gid];
    float sh = shat[gid];
    float si = s[gid];
    float ti = t[gid];
    float xi = x[gid];
    x[gid] = (xi + ((a * ph) + (om * sh)));
    r[gid] = (si - (om * ti));
  }
}
