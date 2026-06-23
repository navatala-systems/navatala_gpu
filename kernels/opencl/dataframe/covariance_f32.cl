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

__kernel void navatala_dataframe_covariance_f32(__global const float* inputX, __global const float* inputY, __global const uint* count, __global const float* meanX, __global const float* meanY, __global float* covariance) {
  int gid0 = (int)get_global_id(0);
  __local float sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint countVal = count[0];
  float meanXVal = meanX[0];
  float meanYVal = meanY[0];
  bool inBounds = (gid < countVal);
  float x = inputX[gid];
  float y = inputY[gid];
  float dx = (x - meanXVal);
  float dy = (y - meanYVal);
  float product = (dx * dy);
  if (inBounds) {
    sdata[lid] = product;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint covF32_reductionStride = (uint)(128u);
  for (int covF32_reductionStep = 0; covF32_reductionStep < (int)(8); ++covF32_reductionStep) {
    uint covF32_stride = covF32_reductionStride;
    if (lid < covF32_stride) {
      float covF32_other = sdata[(lid + covF32_stride)];
      float covF32_mine = sdata[lid];
      float covF32_sum = (covF32_mine + covF32_other);
      sdata[lid] = covF32_sum;
    }
    uint covF32_strideToHalve = covF32_reductionStride;
    uint covF32_nextStride = (covF32_strideToHalve >> (uint)(1u));
    covF32_reductionStride = covF32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    float totalSum = sdata[(uint)(0u)];
    float countFloat = ((float)(countVal));
    float covVal = (totalSum / countFloat);
    covariance[(uint)(0u)] = covVal;
  }
}
