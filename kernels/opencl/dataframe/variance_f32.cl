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

__kernel void navatala_dataframe_variance_f32(__global const float* _input, __global const uint* count, __global const float* mean, __global float* variance) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  float meanVal = mean[0];
  bool inBounds = (gid < countVal);
  float x = _input[gid];
  float diff = (x - meanVal);
  float diffSq = (diff * diff);
  if (inBounds) {
    sdata[lid] = diffSq;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint reductionStride = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint stride = reductionStride;
    if ((lid < stride)) {
      uint partnerIdx = (lid + stride);
      float myVal = sdata[lid];
      float partnerVal = sdata[partnerIdx];
      sdata[lid] = (myVal + partnerVal);
    }
    uint strideToHalve = reductionStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    reductionStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint zeroU32 = ((uint)(0));
  if ((lid == zeroU32)) {
    float totalSumSq = sdata[0];
    float countFloat = ((float)(countVal));
    float varianceVal = (totalSumSq / countFloat);
    variance[0] = varianceVal;
  }
}
