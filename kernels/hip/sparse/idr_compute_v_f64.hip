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
extern "C" __global__ void navatala_sparse_idr_compute_v_f64(const double* r, const double* dR, const double* c, const unsigned int* n, const unsigned int* s, double* v) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  if ((gid < N)) {
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)(S); ++k) {
      double ck = c[k];
      double dr = dR[(gid + (k * N))];
      acc = (acc + (ck * dr));
    }
    double ri = r[gid];
    v[gid] = (ri - acc);
  }
}
