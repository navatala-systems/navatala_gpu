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
extern "C" __global__ void navatala_sparse_gmres_givens_rotation_f64(const double* cs, const double* sn, const unsigned int* colIdx, double* h, double* csNew, double* snNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int ci = ((int)(colIdx[0]));
  for (int k = 0; k < (int)(ci); ++k) {
    double csk = cs[k];
    double snk = sn[k];
    double hk = h[k];
    double hk1 = h[(k + 1)];
    h[k] = ((csk * hk) + (snk * hk1));
    h[(k + 1)] = ((csk * hk1) - (snk * hk));
  }
  double hj = h[ci];
  double hj1 = h[(ci + 1)];
  double r = sqrt(((hj * hj) + (hj1 * hj1)));
  double csVal = (hj / r);
  double snVal = (hj1 / r);
  csNew[0] = csVal;
  snNew[0] = snVal;
  h[ci] = r;
  h[(ci + 1)] = __longlong_as_double(0x0000000000000000ull);
}
