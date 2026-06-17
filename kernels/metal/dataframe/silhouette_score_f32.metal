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

kernel void navatala_dataframe_silhouette_score_f32(device const float* distMatrix [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* clusterCounts [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* silhouette [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = labels[gid];
    uint myClusterCount = clusterCounts[myCluster];
    uint rowBase = (gid * nVal);
    float intraSumAccum = as_type<float>(0x00000000u);
    uint intraCountAccum = 0u;
    for (int j = 0; j < (int)(nVal); ++j) {
      uint jU32 = ((uint)(j));
      uint otherCluster = labels[jU32];
      bool sameCluster = (otherCluster == myCluster);
      bool notSelf = (gid != jU32);
      bool shouldInclude = (sameCluster && notSelf);
      if (shouldInclude) {
        uint distIdx = (rowBase + jU32);
        float dist = distMatrix[distIdx];
        float curIntraSum = intraSumAccum;
        float newIntraSum = (curIntraSum + dist);
        intraSumAccum = newIntraSum;
        uint curIntraCount = intraCountAccum;
        uint newIntraCount = (curIntraCount + 1u);
        intraCountAccum = newIntraCount;
      }
    }
    float intraSum = intraSumAccum;
    uint intraCount = intraCountAccum;
    bool hasIntra = (intraCount > 0u);
    float intraCountF = ((float)(intraCount));
    float aVal = ((hasIntra) ? ((intraSum / intraCountF)) : (as_type<float>(0x00000000u)));
    float bValAccum = as_type<float>(0x7f7fffffu);
    float interSumAccum = as_type<float>(0x00000000u);
    uint interCountAccum = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      uint cU32 = ((uint)(c));
      bool isOtherCluster = (cU32 != myCluster);
      if (isOtherCluster) {
        interSumAccum = as_type<float>(0x00000000u);
        interCountAccum = 0u;
        for (int j = 0; j < (int)(nVal); ++j) {
          uint jU32 = ((uint)(j));
          uint otherClusterJ = labels[jU32];
          bool inTargetCluster = (otherClusterJ == cU32);
          if (inTargetCluster) {
            uint distIdx = (rowBase + jU32);
            float dist = distMatrix[distIdx];
            float curInterSum = interSumAccum;
            float newInterSum = (curInterSum + dist);
            interSumAccum = newInterSum;
            uint curInterCount = interCountAccum;
            uint newInterCount = (curInterCount + 1u);
            interCountAccum = newInterCount;
          }
        }
        float interSum = interSumAccum;
        uint interCount = interCountAccum;
        bool hasInter = (interCount > 0u);
        if (hasInter) {
          float interCountF = ((float)(interCount));
          float meanDist = (interSum / interCountF);
          float curBVal = bValAccum;
          bool isBetter = (meanDist < curBVal);
          float newBVal = ((isBetter) ? (meanDist) : (curBVal));
          bValAccum = newBVal;
        }
      }
    }
    float bVal = bValAccum;
    float maxAB = (((aVal > bVal)) ? (aVal) : (bVal));
    float diff = (bVal - aVal);
    bool maxIsZero = (maxAB == as_type<float>(0x00000000u));
    float score = ((maxIsZero) ? (as_type<float>(0x00000000u)) : ((diff / maxAB)));
    silhouette[gid] = score;
  }
}
