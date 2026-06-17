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

kernel void navatala_ml_count_clusters_f32(device const int* labels [[buffer(0)]], device const uint* nPoints [[buffer(1)]], device int* numClusters [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup int sdata[256];
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  int myLabel = ((inBounds) ? (labels[gid]) : (-1));
  sdata[lid] = myLabel;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint countClstF32RedStride = 128u;
  for (int countClstF32RedStep = 0; countClstF32RedStep < (int)(8); ++countClstF32RedStep) {
    uint countClstF32Stride = countClstF32RedStride;
    if ((lid < countClstF32Stride)) {
      uint otherIdx = (lid + countClstF32Stride);
      int other = sdata[otherIdx];
      int mine = sdata[lid];
      bool isGreater = (other > mine);
      int maxVal = ((isGreater) ? (other) : (mine));
      sdata[lid] = maxVal;
    }
    uint countClstF32StrideToHalve = countClstF32RedStride;
    uint countClstF32NextStride = (countClstF32StrideToHalve >> 1u);
    countClstF32RedStride = countClstF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    int maxLabel = sdata[0];
    int count = (maxLabel + 1);
    numClusters[0] = count;
  }
}
