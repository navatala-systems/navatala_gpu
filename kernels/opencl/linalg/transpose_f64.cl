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
__kernel void navatala_linalg_transpose_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* B) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[0];
  uint cols = n[0];
  uint totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / cols);
    uint j = (gid % cols);
    uint srcIdx = ((i * cols) + j);
    uint dstIdx = ((j * rows) + i);
    double val = A[srcIdx];
    B[dstIdx] = val;
  }
}
