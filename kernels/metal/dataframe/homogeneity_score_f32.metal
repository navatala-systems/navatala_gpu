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

kernel void navatala_dataframe_homogeneity_score_f32(device const float* contingency [[buffer(0)]], device const float* rowSums [[buffer(1)]], device const float* colSums [[buffer(2)]], device const float* n [[buffer(3)]], device const uint* numTrue [[buffer(4)]], device const uint* numPred [[buffer(5)]], device float* homogeneity [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float hycSum[256];
  threadgroup float hySum[256];
  float nVal = n[0u];
  uint numTrueVal = numTrue[0u];
  uint numPredVal = numPred[0u];
  uint totalCells = (numTrueVal * numPredVal);
  float localHYCAccum = as_type<float>(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      uint j = (idxU32 % numPredVal);
      float nij = contingency[idxU32];
      float bj = colSums[j];
      bool nijGtZero = (nij > as_type<float>(0x00000000u));
      bool bjGtZero = (bj > as_type<float>(0x00000000u));
      bool bothGtZero = (nijGtZero && bjGtZero);
      if (bothGtZero) {
        float pij = (nij / nVal);
        float condProb = (nij / bj);
        float logCondProb = log(condProb);
        float termHYC = (pij * logCondProb);
        float curHYC = localHYCAccum;
        float newHYC = (curHYC - termHYC);
        localHYCAccum = newHYC;
      }
    }
  }
  float localHYC = localHYCAccum;
  float localHYAccumHom = as_type<float>(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      float ai = rowSums[iU32];
      bool aiGtZero = (ai > as_type<float>(0x00000000u));
      if (aiGtZero) {
        float pi = (ai / nVal);
        float logPi = log(pi);
        float termHY = (pi * logPi);
        float curHY = localHYAccumHom;
        float newHY = (curHY - termHY);
        localHYAccumHom = newHY;
      }
    }
  }
  float localHY = localHYAccumHom;
  hycSum[lid] = localHYC;
  hySum[lid] = localHY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int stride = 0; stride < (int)(128u); ++stride) {
    uint strideU32 = ((uint)(stride));
    if (lid < strideU32) {
      float otherHYC = hycSum[(lid + strideU32)];
      float mineHYC = hycSum[lid];
      hycSum[lid] = (mineHYC + otherHYC);
      float otherHY = hySum[(lid + strideU32)];
      float mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float hycFinal = hycSum[0u];
    float hyFinal = hySum[0u];
    bool hyIsZero = (hyFinal == as_type<float>(0x00000000u));
    float homoVal = ((hyIsZero) ? (as_type<float>(0x3f800000u)) : ((as_type<float>(0x3f800000u) - (hycFinal / hyFinal))));
    homogeneity[0u] = homoVal;
  }
}
