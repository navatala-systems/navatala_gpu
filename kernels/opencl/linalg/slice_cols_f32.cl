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

__kernel void navatala_linalg_slice_cols_f32(__global const float* A, __global const uint* colStart, __global const uint* colEnd, __global const uint* numRows, __global const uint* srcCols, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint cs = colStart[(uint)(0u)];
  uint ce = colEnd[(uint)(0u)];
  uint m = numRows[(uint)(0u)];
  uint n = srcCols[(uint)(0u)];
  uint outCols = (ce - cs);
  uint totalElems = (m * outCols);
  if (idx < totalElems) {
    uint outRow = (idx / outCols);
    uint outCol = (idx % outCols);
    uint srcCol = (cs + outCol);
    uint srcIdx = ((outRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}
