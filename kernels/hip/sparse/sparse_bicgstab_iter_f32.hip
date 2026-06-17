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
extern "C" __global__ void navatala_sparse_sparse_bicgstab_iter_f32(const float* x, const float* p, const float* r, const float* v, float* s, const float* t, const float* alpha, const float* omega, const unsigned int* n, float* x_new, float* r_new) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int i = gid;
  unsigned int nVal = n[0u];
  if ((i < nVal)) {
    float alphaVal = alpha[0u];
    float omegaVal = omega[0u];
    float xi = x[i];
    float pi = p[i];
    float si = s[i];
    float ti = t[i];
    float alphaPi = (alphaVal * pi);
    float omegaSi = (omegaVal * si);
    float xNew = (xi + alphaPi);
    float xNew2 = (xNew + omegaSi);
    x_new[i] = xNew2;
    float omegaTi = (omegaVal * ti);
    float rNew = (si - omegaTi);
    r_new[i] = rNew;
  }
}
