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

kernel void navatala_ml_compute_gcv_score_f32(device const float* y [[buffer(0)]], device const float* yPred [[buffer(1)]], device const float* traceH [[buffer(2)]], device const uint* n [[buffer(3)]], device float* gcv [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  float traceHVal = traceH[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float residual = (yVal - yPredVal);
    float residualSq = (residual * residual);
    sdata[lid] = residualSq;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint gcv32_reductionStride = 128u;
  for (int gcv32_reductionStep = 0; gcv32_reductionStep < (int)(8); ++gcv32_reductionStep) {
    uint gcv32_stride = gcv32_reductionStride;
    if ((lid < gcv32_stride)) {
      float gcv32_other = sdata[(lid + gcv32_stride)];
      float gcv32_mine = sdata[lid];
      float gcv32_sum = (gcv32_mine + gcv32_other);
      sdata[lid] = gcv32_sum;
    }
    uint gcv32_strideToHalve = gcv32_reductionStride;
    uint gcv32_nextStride = (gcv32_strideToHalve >> 1u);
    gcv32_reductionStride = gcv32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float rss = sdata[0];
    float nFloat = ((float)(nVal));
    float traceHOverN = (traceHVal / nFloat);
    float oneMinusRatio = (as_type<float>(0x3f800000u) - traceHOverN);
    float denom = (oneMinusRatio * oneMinusRatio);
    float fullDenom = (nFloat * denom);
    float gcvVal = (rss / fullDenom);
    gcv[0] = gcvVal;
  }
}
