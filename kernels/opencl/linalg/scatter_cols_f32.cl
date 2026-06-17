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

__kernel void navatala_linalg_scatter_cols_f32(__global const float* src, __global const uint* indices, __global const uint* m, __global const uint* k, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint mVal = m[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * kVal);
  if ((tid < totalElems)) {
    uint row = (tid / kVal);
    uint srcColIdx = (tid % kVal);
    uint dstCol = indices[srcColIdx];
    uint dstIdx = ((row * nVal) + dstCol);
    float val = src[tid];
    A[dstIdx] = val;
  }
}
