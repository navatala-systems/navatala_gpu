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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_mcd_compute_center_f32(device const float* dataX [[buffer(0)]], device const float* dataY [[buffer(1)]], device const uint* mask [[buffer(2)]], device const uint* n [[buffer(3)]], device float* robustMeanX [[buffer(4)]], device float* robustMeanY [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdataX[256];
  threadgroup float sdataY[256];
  threadgroup uint scount[256];
  uint len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    uint m = mask[gid];
    float x = dataX[gid];
    float y = dataY[gid];
    float mF = ((float)(m));
    float maskedX = (x * mF);
    float maskedY = (y * mF);
    sdataX[lid] = maskedX;
    sdataY[lid] = maskedY;
    scount[lid] = m;
  } else {
    sdataX[lid] = as_type<float>(0x00000000u);
    sdataY[lid] = as_type<float>(0x00000000u);
    scount[lid] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint ctr1ReductionStride = 128u;
  for (int ctr1ReductionStep = 0; ctr1ReductionStep < (int)(8); ++ctr1ReductionStep) {
    uint ctr1Stride = ctr1ReductionStride;
    if ((lid < ctr1Stride)) {
      float otherX = sdataX[(lid + ctr1Stride)];
      float otherY = sdataY[(lid + ctr1Stride)];
      uint otherC = scount[(lid + ctr1Stride)];
      float mineX = sdataX[lid];
      float mineY = sdataY[lid];
      uint mineC = scount[lid];
      float sumX = (mineX + otherX);
      float sumY = (mineY + otherY);
      uint sumC = (mineC + otherC);
      sdataX[lid] = sumX;
      sdataY[lid] = sumY;
      scount[lid] = sumC;
    }
    uint ctr1StrideToHalve = ctr1ReductionStride;
    uint ctr1NextStride = (ctr1StrideToHalve >> 1u);
    ctr1ReductionStride = ctr1NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float totalX = sdataX[0];
    float totalY = sdataY[0];
    uint totalCount = scount[0];
    float countF = ((float)(totalCount));
    float safeCount = (((countF > as_type<float>(0x00000000u))) ? (countF) : (as_type<float>(0x3f800000u)));
    float meanX = (totalX / safeCount);
    float meanY = (totalY / safeCount);
    robustMeanX[0] = meanX;
    robustMeanY[0] = meanY;
  }
}
