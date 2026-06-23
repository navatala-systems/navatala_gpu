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
extern "C" __global__ void navatala_sparse_jacobi_davidson_correction_f32(const float* r, const float* v, const float* theta, const unsigned int* n, float* t) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float ri = r[gid];
    float th = theta[0];
    float negR = (-ri);
    float absTh = abs(th);
    float safeTh = (((absTh > __uint_as_float(0x0da24260u))) ? (th) : (__uint_as_float(0x3f800000u)));
    bool useDiag = (absTh > __uint_as_float(0x0da24260u));
    float ti = ((useDiag) ? ((negR / safeTh)) : (negR));
    t[gid] = ti;
  }
}
