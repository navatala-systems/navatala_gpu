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
__kernel void navatala_sparse_csr_permute_cols_f64(__global const uint* colIdx, __global const double* values, __global const uint* perm, __global const uint* nnz, __global uint* newColIdx, __global double* newValues) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nnz[0]));
  if ((gid < N)) {
    uint c = colIdx[gid];
    uint newC = perm[((int)(c))];
    newColIdx[gid] = newC;
    newValues[gid] = values[gid];
  }
}
