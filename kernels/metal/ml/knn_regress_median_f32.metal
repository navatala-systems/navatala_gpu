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

kernel void navatala_ml_knn_regress_median_f32(device const uint* neighborIndices [[buffer(0)]], device const float* targetValues [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* k [[buffer(3)]], device float* predictions [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    threadgroup float sortBuf[256];
    uint baseIdx = (gid * kVal);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      sortBuf[jU32] = targetVal;
    }
    uint jIdxAccum = 0u;
    for (int i = 0; i < (int)(kVal); ++i) {
      uint iU32 = ((uint)(i));
      uint zeroU32 = ((uint)(0));
      if ((iU32 > zeroU32)) {
        float key = sortBuf[iU32];
        uint initialJIdx = (iU32 - 1u);
        jIdxAccum = initialJIdx;
        for (int iter = 0; iter < (int)(iU32); ++iter) {
          uint currJIdx = jIdxAccum;
          float currentVal = sortBuf[currJIdx];
          bool shouldShift = (currentVal > key);
          if (shouldShift) {
            uint jPlus1 = (currJIdx + 1u);
            sortBuf[jPlus1] = currentVal;
            uint newJIdx = (((currJIdx == 0u)) ? (0u) : ((currJIdx - 1u)));
            jIdxAccum = newJIdx;
          }
        }
        uint finalJIdx = jIdxAccum;
        float lastCurrentVal = sortBuf[finalJIdx];
        uint insertPos = (((lastCurrentVal > key)) ? (finalJIdx) : ((finalJIdx + 1u)));
        sortBuf[insertPos] = key;
      }
    }
    uint midIdx = (kVal / 2u);
    uint kMod2 = (kVal % 2u);
    bool isOdd = (kMod2 != 0u);
    float midVal = sortBuf[midIdx];
    uint midMinus1Idx = (((midIdx == 0u)) ? (0u) : ((midIdx - 1u)));
    float midMinus1Val = sortBuf[midMinus1Idx];
    float evenMedian = ((midMinus1Val + midVal) / as_type<float>(0x40000000u));
    float median = ((isOdd) ? (midVal) : (evenMedian));
    predictions[gid] = median;
  }
}
