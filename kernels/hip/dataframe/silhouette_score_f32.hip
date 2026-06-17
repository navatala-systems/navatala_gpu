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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_silhouette_score_f32(const float* distMatrix, const unsigned int* labels, const unsigned int* clusterCounts, const unsigned int* n, const unsigned int* k, float* silhouette) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = labels[gid];
    unsigned int myClusterCount = clusterCounts[myCluster];
    unsigned int rowBase = (gid * nVal);
    float intraSumAccum = __uint_as_float(0x00000000u);
    unsigned int intraCountAccum = 0u;
    for (int j = 0; j < (int)(nVal); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int otherCluster = labels[jU32];
      bool sameCluster = (otherCluster == myCluster);
      bool notSelf = (gid != jU32);
      bool shouldInclude = (sameCluster && notSelf);
      if (shouldInclude) {
        unsigned int distIdx = (rowBase + jU32);
        float dist = distMatrix[distIdx];
        float curIntraSum = intraSumAccum;
        float newIntraSum = (curIntraSum + dist);
        intraSumAccum = newIntraSum;
        unsigned int curIntraCount = intraCountAccum;
        unsigned int newIntraCount = (curIntraCount + 1u);
        intraCountAccum = newIntraCount;
      }
    }
    float intraSum = intraSumAccum;
    unsigned int intraCount = intraCountAccum;
    bool hasIntra = (intraCount > 0u);
    float intraCountF = ((float)(intraCount));
    float aVal = ((hasIntra) ? ((intraSum / intraCountF)) : (__uint_as_float(0x00000000u)));
    float bValAccum = __uint_as_float(0x7f7fffffu);
    float interSumAccum = __uint_as_float(0x00000000u);
    unsigned int interCountAccum = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      unsigned int cU32 = ((unsigned int)(c));
      bool isOtherCluster = (cU32 != myCluster);
      if (isOtherCluster) {
        interSumAccum = __uint_as_float(0x00000000u);
        interCountAccum = 0u;
        for (int j = 0; j < (int)(nVal); ++j) {
          unsigned int jU32 = ((unsigned int)(j));
          unsigned int otherClusterJ = labels[jU32];
          bool inTargetCluster = (otherClusterJ == cU32);
          if (inTargetCluster) {
            unsigned int distIdx = (rowBase + jU32);
            float dist = distMatrix[distIdx];
            float curInterSum = interSumAccum;
            float newInterSum = (curInterSum + dist);
            interSumAccum = newInterSum;
            unsigned int curInterCount = interCountAccum;
            unsigned int newInterCount = (curInterCount + 1u);
            interCountAccum = newInterCount;
          }
        }
        float interSum = interSumAccum;
        unsigned int interCount = interCountAccum;
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
    bool maxIsZero = (maxAB == __uint_as_float(0x00000000u));
    float score = ((maxIsZero) ? (__uint_as_float(0x00000000u)) : ((diff / maxAB)));
    silhouette[gid] = score;
  }
}
