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

__kernel void navatala_dataframe_silhouette_score_f32(__global const float* distMatrix, __global const uint* labels, __global const uint* clusterCounts, __global const uint* n, __global const uint* k, __global float* silhouette) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = labels[gid];
    uint myClusterCount = clusterCounts[myCluster];
    uint rowBase = (gid * nVal);
    float intraSumAccum = as_float(0x00000000u);
    uint intraCountAccum = (uint)(0u);
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
        uint newIntraCount = (curIntraCount + (uint)(1u));
        intraCountAccum = newIntraCount;
      }
    }
    float intraSum = intraSumAccum;
    uint intraCount = intraCountAccum;
    bool hasIntra = (intraCount > (uint)(0u));
    float intraCountF = ((float)(intraCount));
    float aVal = ((hasIntra) ? ((intraSum / intraCountF)) : (as_float(0x00000000u)));
    float bValAccum = as_float(0x7f7fffffu);
    float interSumAccum = as_float(0x00000000u);
    uint interCountAccum = (uint)(0u);
    for (int c = 0; c < (int)(kVal); ++c) {
      uint cU32 = ((uint)(c));
      bool isOtherCluster = (cU32 != myCluster);
      if (isOtherCluster) {
        interSumAccum = as_float(0x00000000u);
        interCountAccum = (uint)(0u);
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
            uint newInterCount = (curInterCount + (uint)(1u));
            interCountAccum = newInterCount;
          }
        }
        float interSum = interSumAccum;
        uint interCount = interCountAccum;
        bool hasInter = (interCount > (uint)(0u));
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
    bool maxIsZero = (maxAB == as_float(0x00000000u));
    float score = ((maxIsZero) ? (as_float(0x00000000u)) : ((diff / maxAB)));
    silhouette[gid] = score;
  }
}
