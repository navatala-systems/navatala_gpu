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
__kernel void navatala_sparse_coo_transpose_f64(__global const uint* rowInd, __global const uint* colInd, __global const double* values, __global const uint* nnz, __global uint* rowInd_out, __global uint* colInd_out, __global double* values_out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint i = gid;
  if ((i < nnz[(uint)(0u)])) {
    uint col = colInd[i];
    rowInd_out[i] = col;
    uint row = rowInd[i];
    colInd_out[i] = row;
    double val = values[i];
    values_out[i] = val;
  }
}
