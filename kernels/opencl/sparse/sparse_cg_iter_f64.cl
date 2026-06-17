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
__kernel void navatala_sparse_sparse_cg_iter_f64(__global const uint* rowPtr, __global const uint* colInd, __global const double* values, __global const double* x, __global const double* p, __global const double* r, __global double* q, __global const double* rTr, __global const double* alpha, __global const uint* n, __global double* x_new, __global double* r_new) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint i = gid;
  uint nVal = n[(uint)(0u)];
  if ((i < nVal)) {
    double alphaVal = alpha[(uint)(0u)];
    double xi = x[i];
    double pi = p[i];
    double ri = r[i];
    double qi = q[i];
    double alphaPi = (alphaVal * pi);
    double xNew = (xi + alphaPi);
    x_new[i] = xNew;
    double alphaQi = (alphaVal * qi);
    double rNew = (ri - alphaQi);
    r_new[i] = rNew;
  }
}
