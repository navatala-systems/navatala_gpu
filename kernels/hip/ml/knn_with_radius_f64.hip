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
extern "C" __global__ void navatala_ml_knn_with_radius_f64(const double* queries, const double* data, const unsigned int* q, const unsigned int* n, const unsigned int* d, const unsigned int* k, const double* radius, unsigned int* indices, double* distances, unsigned int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int qVal = q[0];
  unsigned int nVal = n[0];
  unsigned int dVal = d[0];
  unsigned int kVal = k[0];
  double radiusVal = radius[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    unsigned int queryBase = (gid * dVal);
    unsigned int outBase = (gid * kVal);
    unsigned int neighborCountAccum = 0u;
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      unsigned int initIdxU32 = ((unsigned int)(initIdx));
      unsigned int outIdx = (outBase + initIdxU32);
      distances[outIdx] = __longlong_as_double(0x7fee42d130773b76ull);
      indices[outIdx] = 4294967295u;
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      unsigned int dataIdxU32 = ((unsigned int)(dataIdx));
      unsigned int dataBase = (dataIdxU32 * dVal);
      double distSumAccum = __longlong_as_double(0x0000000000000000ull);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        unsigned int dimIdxU32 = ((unsigned int)(dimIdx));
        unsigned int qIdx = (queryBase + dimIdxU32);
        unsigned int dIdx = (dataBase + dimIdxU32);
        double qv = queries[qIdx];
        double dv = data[dIdx];
        double diff = (qv - dv);
        double diffSq = (diff * diff);
        double currDistSum = distSumAccum;
        double newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      double distSum = distSumAccum;
      bool withinRadius = (distSum <= radiusVal);
      if (withinRadius) {
        unsigned int worstIdx = (outBase + (kVal - 1u));
        double worstDist = distances[worstIdx];
        bool shouldInsert = (distSum < worstDist);
        if (shouldInsert) {
          unsigned int currNeighborCount = neighborCountAccum;
          unsigned int countPlusOne = (currNeighborCount + 1u);
          bool countCapped = (currNeighborCount < kVal);
          unsigned int newNeighborCount = ((countCapped) ? (countPlusOne) : (kVal));
          neighborCountAccum = newNeighborCount;
          unsigned int insertPosAccum = (kVal - 1u);
          for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
            unsigned int scanIdxU32 = ((unsigned int)(scanIdx));
            unsigned int checkPos = ((kVal - 1u) - scanIdxU32);
            unsigned int checkIdx = (outBase + checkPos);
            double checkDist = distances[checkIdx];
            bool shouldShift = (distSum < checkDist);
            unsigned int currInsertPos = insertPosAccum;
            unsigned int newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
            insertPosAccum = newInsertPos;
          }
          unsigned int insertPos = insertPosAccum;
          for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
            unsigned int shiftIdxU32 = ((unsigned int)(shiftIdx));
            unsigned int shiftPos = ((kVal - 2u) - shiftIdxU32);
            bool shouldDoShift = (shiftPos >= insertPos);
            if (shouldDoShift) {
              unsigned int srcIdx = (outBase + shiftPos);
              unsigned int dstIdx = (outBase + (shiftPos + 1u));
              double shiftDist = distances[srcIdx];
              unsigned int shiftIndex = indices[srcIdx];
              distances[dstIdx] = shiftDist;
              indices[dstIdx] = shiftIndex;
            }
          }
          unsigned int insertIdx = (outBase + insertPos);
          distances[insertIdx] = distSum;
          indices[insertIdx] = dataIdxU32;
        }
      }
    }
    unsigned int neighborCount = neighborCountAccum;
    counts[gid] = neighborCount;
  }
}
