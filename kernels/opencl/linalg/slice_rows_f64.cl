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
__kernel void navatala_linalg_slice_rows_f64(__global const double* A, __global const uint* rowStart, __global const uint* rowEnd, __global const uint* numCols, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint rs = rowStart[(uint)(0u)];
  uint re = rowEnd[(uint)(0u)];
  uint n = numCols[(uint)(0u)];
  uint numRows = (re - rs);
  uint totalElems = (numRows * n);
  if (idx < totalElems) {
    uint outRow = (idx / n);
    uint outCol = (idx % n);
    uint srcRow = (rs + outRow);
    uint srcIdx = ((srcRow * n) + outCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}
