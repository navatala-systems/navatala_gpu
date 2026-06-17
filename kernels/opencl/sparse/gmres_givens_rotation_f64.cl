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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_gmres_givens_rotation_f64(__global const double* cs, __global const double* sn, __global const uint* colIdx, __global double* h, __global double* csNew, __global double* snNew) {
  int gid0 = (int)get_global_id(0);
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
  h[(ci + 1)] = as_double(0x0000000000000000ul);
}
