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

__kernel void navatala_linalg_norm_linf_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  float val = fabs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint linfF32RedStride = (uint)(128u);
  for (int linfF32RedStep = 0; linfF32RedStep < (int)(8); ++linfF32RedStep) {
    uint linfF32Stride = linfF32RedStride;
    if (lid < linfF32Stride) {
      float other = sdata[(lid + linfF32Stride)];
      float mine = sdata[lid];
      bool mineGtOther = (mine > other);
      float maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint linfF32StrideToHalve = linfF32RedStride;
    uint linfF32NextStride = (linfF32StrideToHalve >> (uint)(1u));
    linfF32RedStride = linfF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    result[0] = sdata[0];
  }
}
