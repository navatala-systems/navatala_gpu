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
__kernel void navatala_sparse_gmres_arnoldi_batch_project_f64(__global const double* V, __global const double* hCol, __global const uint* n, __global const uint* j, __global double* w) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  int J = ((int)(j[0]));
  if ((gid < N)) {
    double acc = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(J); ++k) {
      double hk = hCol[k];
      double vik = V[(gid + (k * N))];
      acc = (acc + (hk * vik));
    }
    double wi = w[gid];
    w[gid] = (wi - acc);
  }
}
