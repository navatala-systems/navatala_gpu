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

__kernel void navatala_graph_normalize_eigenvector_rows_f32(__global const float* V, __global const uint* n, __global const uint* k, __global float* V_norm) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local float sdata[256];
  uint nVal = n[0];
  uint kVal = k[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      float vVal = V[elemIdx];
      float vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint normF32RedStride = (uint)(128u);
    for (int normF32RedStep = 0; normF32RedStep < (int)(8); ++normF32RedStep) {
      uint normF32Stride = normF32RedStride;
      if (lid < normF32Stride) {
        float other = sdata[(lid + normF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint normF32StrideToHalve = normF32RedStride;
      uint normF32NextStride = (normF32StrideToHalve >> (uint)(1u));
      normF32RedStride = normF32NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == (uint)(0u)) {
      float sumSq = sdata[0];
      float normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (colInBounds) {
      float norm = sdata[0];
      float vVal2 = V[elemIdx];
      float eps = as_float(0x2edbe6ffu);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = as_float(0x00000000u);
      } else {
        float normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}
