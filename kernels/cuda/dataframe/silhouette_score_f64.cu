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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_silhouette_score_f64(const double* distMatrix, const unsigned int* labels, const unsigned int* clusterCounts, const unsigned int* n, const unsigned int* k, double* silhouette) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = labels[gid];
    unsigned int myClusterCount = clusterCounts[myCluster];
    unsigned int rowBase = (gid * nVal);
    double intraSumAccum64 = __longlong_as_double(0x0000000000000000ull);
    unsigned int intraCountAccum64 = 0u;
    for (int j = 0; j < (int)(nVal); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int otherCluster = labels[jU32];
      bool sameCluster = (otherCluster == myCluster);
      bool notSelf = (gid != jU32);
      bool shouldInclude = (sameCluster && notSelf);
      if (shouldInclude) {
        unsigned int distIdx = (rowBase + jU32);
        double dist = distMatrix[distIdx];
        double curIntraSum = intraSumAccum64;
        double newIntraSum = (curIntraSum + dist);
        intraSumAccum64 = newIntraSum;
        unsigned int curIntraCount = intraCountAccum64;
        unsigned int newIntraCount = (curIntraCount + 1u);
        intraCountAccum64 = newIntraCount;
      }
    }
    double intraSum = intraSumAccum64;
    unsigned int intraCount = intraCountAccum64;
    bool hasIntra = (intraCount > 0u);
    double intraCountF = ((double)(intraCount));
    double aVal = ((hasIntra) ? ((intraSum / intraCountF)) : (__longlong_as_double(0x0000000000000000ull)));
    double bValAccum64 = __longlong_as_double(0x7fefffffffffffffull);
    double interSumAccum64 = __longlong_as_double(0x0000000000000000ull);
    unsigned int interCountAccum64 = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      unsigned int cU32 = ((unsigned int)(c));
      bool isOtherCluster = (cU32 != myCluster);
      if (isOtherCluster) {
        interSumAccum64 = __longlong_as_double(0x0000000000000000ull);
        interCountAccum64 = 0u;
        for (int j = 0; j < (int)(nVal); ++j) {
          unsigned int jU32 = ((unsigned int)(j));
          unsigned int otherClusterJ = labels[jU32];
          bool inTargetCluster = (otherClusterJ == cU32);
          if (inTargetCluster) {
            unsigned int distIdx = (rowBase + jU32);
            double dist = distMatrix[distIdx];
            double curInterSum = interSumAccum64;
            double newInterSum = (curInterSum + dist);
            interSumAccum64 = newInterSum;
            unsigned int curInterCount = interCountAccum64;
            unsigned int newInterCount = (curInterCount + 1u);
            interCountAccum64 = newInterCount;
          }
        }
        double interSum = interSumAccum64;
        unsigned int interCount = interCountAccum64;
        bool hasInter = (interCount > 0u);
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
    bool maxIsZero = (maxAB == __longlong_as_double(0x0000000000000000ull));
    double score = ((maxIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : ((diff / maxAB)));
    silhouette[gid] = score;
  }
}
