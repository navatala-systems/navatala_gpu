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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_linalg_reduce_max_f16(__global const half* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = as_float(0xff7ffffdu);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint maxF16ReductionStride = (uint)(128u);
  for (int maxF16ReductionStep = 0; maxF16ReductionStep < (int)(8); ++maxF16ReductionStep) {
    uint maxF16Stride = maxF16ReductionStride;
    if ((lid < maxF16Stride)) {
      float other = sdata[(lid + maxF16Stride)];
      float mine = sdata[lid];
      bool isGreater = (mine > other);
      float maxVal = ((isGreater) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint maxF16StrideToHalve = maxF16ReductionStride;
    uint maxF16NextStride = (maxF16StrideToHalve >> (uint)(1u));
    maxF16ReductionStride = maxF16NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float maxResult = sdata[0];
    result[0] = maxResult;
  }
}
