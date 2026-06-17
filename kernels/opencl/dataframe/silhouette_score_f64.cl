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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_dataframe_silhouette_score_f64(__global const double* distMatrix, __global const uint* labels, __global const uint* clusterCounts, __global const uint* n, __global const uint* k, __global double* silhouette) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = labels[gid];
    uint myClusterCount = clusterCounts[myCluster];
    uint rowBase = (gid * nVal);
    double intraSumAccum64 = as_double(0x0000000000000000ul);
    uint intraCountAccum64 = (uint)(0u);
    for (int j = 0; j < (int)(nVal); ++j) {
      uint jU32 = ((uint)(j));
      uint otherCluster = labels[jU32];
      bool sameCluster = (otherCluster == myCluster);
      bool notSelf = (gid != jU32);
      bool shouldInclude = (sameCluster && notSelf);
      if (shouldInclude) {
        uint distIdx = (rowBase + jU32);
        double dist = distMatrix[distIdx];
        double curIntraSum = intraSumAccum64;
        double newIntraSum = (curIntraSum + dist);
        intraSumAccum64 = newIntraSum;
        uint curIntraCount = intraCountAccum64;
        uint newIntraCount = (curIntraCount + (uint)(1u));
        intraCountAccum64 = newIntraCount;
      }
    }
    double intraSum = intraSumAccum64;
    uint intraCount = intraCountAccum64;
    bool hasIntra = (intraCount > (uint)(0u));
    double intraCountF = ((double)(intraCount));
    double aVal = ((hasIntra) ? ((intraSum / intraCountF)) : (as_double(0x0000000000000000ul)));
    double bValAccum64 = as_double(0x7feffffffffffffful);
    double interSumAccum64 = as_double(0x0000000000000000ul);
    uint interCountAccum64 = (uint)(0u);
    for (int c = 0; c < (int)(kVal); ++c) {
      uint cU32 = ((uint)(c));
      bool isOtherCluster = (cU32 != myCluster);
      if (isOtherCluster) {
        interSumAccum64 = as_double(0x0000000000000000ul);
        interCountAccum64 = (uint)(0u);
        for (int j = 0; j < (int)(nVal); ++j) {
          uint jU32 = ((uint)(j));
          uint otherClusterJ = labels[jU32];
          bool inTargetCluster = (otherClusterJ == cU32);
          if (inTargetCluster) {
            uint distIdx = (rowBase + jU32);
            double dist = distMatrix[distIdx];
            double curInterSum = interSumAccum64;
            double newInterSum = (curInterSum + dist);
            interSumAccum64 = newInterSum;
            uint curInterCount = interCountAccum64;
            uint newInterCount = (curInterCount + (uint)(1u));
            interCountAccum64 = newInterCount;
          }
        }
        double interSum = interSumAccum64;
        uint interCount = interCountAccum64;
        bool hasInter = (interCount > (uint)(0u));
        if (hasInter) {
          double interCountF = ((double)(interCount));
          double meanDist = (interSum / interCountF);
          double curBVal = bValAccum64;
          bool isBetter = (meanDist < curBVal);
          double newBVal = ((isBetter) ? (meanDist) : (curBVal));
          bValAccum64 = newBVal;
        }
      }
    }
    double bVal = bValAccum64;
    double maxAB = (((aVal > bVal)) ? (aVal) : (bVal));
    double diff = (bVal - aVal);
    bool maxIsZero = (maxAB == as_double(0x0000000000000000ul));
    double score = ((maxIsZero) ? (as_double(0x0000000000000000ul)) : ((diff / maxAB)));
    silhouette[gid] = score;
  }
}
