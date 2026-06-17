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

kernel void navatala_ml_explained_variance_ratio_f32(device const float* eigenvalues [[buffer(0)]], device const uint* nFeatures [[buffer(1)]], device const uint* nComponents [[buffer(2)]], device float* ratios [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  bool inBoundsLoad = (gid < d);
  if (inBoundsLoad) {
    float eigVal = eigenvalues[gid];
    sdata[lid] = eigVal;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varRatioF32RedStride = 128u;
  for (int varRatioF32RedStep = 0; varRatioF32RedStep < (int)(8); ++varRatioF32RedStep) {
    uint varRatioF32Stride = varRatioF32RedStride;
    if ((lid < varRatioF32Stride)) {
      float other = sdata[(lid + varRatioF32Stride)];
      float mine = sdata[lid];
      float sumVal = (mine + other);
      sdata[lid] = sumVal;
    }
    uint varRatioF32StrideToHalve = varRatioF32RedStride;
    uint varRatioF32NextStride = (varRatioF32StrideToHalve >> 1u);
    varRatioF32RedStride = varRatioF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float totalVariance = sdata[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool inBoundsRatio = (gid < kComp);
  if (inBoundsRatio) {
    float eigVal = eigenvalues[gid];
    float ratio = (eigVal / totalVariance);
    ratios[gid] = ratio;
  }
}
