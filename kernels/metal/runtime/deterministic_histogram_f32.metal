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

kernel void navatala_runtime_deterministic_histogram_f32(device const float* values [[buffer(0)]], device const uint* numBins [[buffer(1)]], device const float* minVal [[buffer(2)]], device const float* maxVal [[buffer(3)]], device const uint* count [[buffer(4)]], device uint* histogram [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup uint binIndices[256];
  threadgroup uint localHist[256];
  uint countVal = count[0];
  uint numBinsVal = numBins[0];
  float minValF = minVal[0];
  float maxValF = maxVal[0];
  float range = (maxValF - minValF);
  float numBinsFloat = ((float)(numBinsVal));
  float binWidth = (range / numBinsFloat);
  if (lid < numBinsVal) {
    localHist[lid] = 0u;
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = values[gid];
    float normalized = ((val - minValF) / binWidth);
    float binIdxFloat = normalized;
    uint binIdx = ((uint)(binIdxFloat));
    uint maxBinIdx = (numBinsVal - 1u);
    uint clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < numBinsVal) {
    uint myBin = lid;
    uint histF32RedStride = 128u;
    for (int histF32RedStep = 0; histF32RedStep < (int)(8); ++histF32RedStep) {
      uint histF32Stride = histF32RedStride;
      if (lid < histF32Stride) {
        uint partnerIdx = (lid + histF32Stride);
        uint storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          uint oldCount = localHist[lid];
          uint newCount = (oldCount + 1u);
          localHist[lid] = newCount;
        }
      }
      uint histF32StrideToHalve = histF32RedStride;
      uint histF32NextStride = (histF32StrideToHalve >> 1u);
      histF32RedStride = histF32NextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      uint binU32 = ((uint)(bin));
      uint localCount = localHist[binU32];
      uint globalCount = histogram[binU32];
      uint newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}
