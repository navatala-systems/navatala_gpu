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
extern "C" __global__ void navatala_sparse_deflated_project_f64(const double* x, const double* Z, const double* ZtX, const unsigned int* n, const unsigned int* k, double* w) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((gid < N)) {
    double xi = x[gid];
    double proj = __longlong_as_double(0x0000000000000000ull);
    int kVal = ((int)(k[0]));
    for (int j = 0; j < (int)(kVal); ++j) {
      double zVal = Z[((j * N) + gid)];
      double zx = ZtX[j];
      proj = (proj + (zVal * zx));
    }
    w[gid] = (xi - proj);
  }
}
