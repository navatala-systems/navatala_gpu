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
__kernel void navatala_sparse_ldu_diagonal_fill_f64(__global const double* diagonal, __global const uint* rowPtr, __global const uint* nCells, __global uint* colInd, __global double* values) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nCells[0]));
  if ((row < N)) {
    int rEnd = ((int)(rowPtr[(row + 1)]));
    int dPos = (rEnd - 1);
    colInd[dPos] = ((uint)(row));
    values[dPos] = diagonal[row];
  }
}
