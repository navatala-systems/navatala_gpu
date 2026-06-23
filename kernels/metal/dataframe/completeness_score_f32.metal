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

kernel void navatala_dataframe_completeness_score_f32(device const float* contingency [[buffer(0)]], device const float* rowSums [[buffer(1)]], device const float* colSums [[buffer(2)]], device const float* n [[buffer(3)]], device const uint* numTrue [[buffer(4)]], device const uint* numPred [[buffer(5)]], device float* completeness [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float hcySum[256];
  threadgroup float hcSum[256];
  float nVal = n[0u];
  uint numTrueVal = numTrue[0u];
  uint numPredVal = numPred[0u];
  uint totalCells = (numTrueVal * numPredVal);
  float localHCYAccum = as_type<float>(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      uint i = (idxU32 / numPredVal);
      float nij = contingency[idxU32];
      float ai = rowSums[i];
      bool nijGtZero = (nij > as_type<float>(0x00000000u));
      bool aiGtZero = (ai > as_type<float>(0x00000000u));
      bool bothGtZero = (nijGtZero && aiGtZero);
      if (bothGtZero) {
        float pij = (nij / nVal);
        float condProb = (nij / ai);
        float logCondProb = log(condProb);
        float termHCY = (pij * logCondProb);
        float curHCY = localHCYAccum;
        float newHCY = (curHCY - termHCY);
        localHCYAccum = newHCY;
      }
    }
  }
  float localHCY = localHCYAccum;
  float localHCAccumComp = as_type<float>(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      float bj = colSums[jU32];
      bool bjGtZero = (bj > as_type<float>(0x00000000u));
      if (bjGtZero) {
        float pj = (bj / nVal);
        float logPj = log(pj);
        float termHC = (pj * logPj);
        float curHC = localHCAccumComp;
        float newHC = (curHC - termHC);
        localHCAccumComp = newHC;
      }
    }
  }
  float localHC = localHCAccumComp;
  hcySum[lid] = localHCY;
  hcSum[lid] = localHC;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int stride = 0; stride < (int)(128u); ++stride) {
    uint strideU32 = ((uint)(stride));
    if (lid < strideU32) {
      float otherHCY = hcySum[(lid + strideU32)];
      float mineHCY = hcySum[lid];
      hcySum[lid] = (mineHCY + otherHCY);
      float otherHC = hcSum[(lid + strideU32)];
      float mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float hcyFinal = hcySum[0u];
    float hcFinal = hcSum[0u];
    bool hcIsZero = (hcFinal == as_type<float>(0x00000000u));
    float compVal = ((hcIsZero) ? (as_type<float>(0x3f800000u)) : ((as_type<float>(0x3f800000u) - (hcyFinal / hcFinal))));
    completeness[0u] = compVal;
  }
}
