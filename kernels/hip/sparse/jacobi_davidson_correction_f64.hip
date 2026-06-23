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
extern "C" __global__ void navatala_sparse_jacobi_davidson_correction_f64(const double* r, const double* v, const double* theta, const unsigned int* n, double* t) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    double ri = r[gid];
    double th = theta[0];
    double negR = (-ri);
    double absTh = abs(th);
    double safeTh = (((absTh > __longlong_as_double(0x01a56e1fc2f8f359ull))) ? (th) : (__longlong_as_double(0x3ff0000000000000ull)));
    bool useDiag = (absTh > __longlong_as_double(0x01a56e1fc2f8f359ull));
    double ti = ((useDiag) ? ((negR / safeTh)) : (negR));
    t[gid] = ti;
  }
}
