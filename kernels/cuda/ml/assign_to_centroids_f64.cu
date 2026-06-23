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
extern "C" __global__ void navatala_ml_assign_to_centroids_f64(const double* data, const double* centroids, const unsigned int* n, const unsigned int* k, const unsigned int* d, unsigned int* assignments) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int dVal = d[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int dataBase = (gid * dVal);
    unsigned int bestClusterAccum = 0u;
    double bestDistAccum = __longlong_as_double(0x7fefffffffffffffull);
    double distSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int c = 0; c < (int)(kVal); ++c) {
      unsigned int cU32 = ((unsigned int)(c));
      unsigned int centroidBase = (cU32 * dVal);
      distSumAccum = __longlong_as_double(0x0000000000000000ull);
      for (int j = 0; j < (int)(dVal); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        unsigned int dataIdx = (dataBase + jU32);
        unsigned int centIdx = (centroidBase + jU32);
        double dataVal = data[dataIdx];
        double centVal = centroids[centIdx];
        double diff = (dataVal - centVal);
        double diffSq = (diff * diff);
        double currDistSum = distSumAccum;
        double newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      double distSum = distSumAccum;
      double bestDist = bestDistAccum;
      if (distSum < bestDist) {
        bestDistAccum = distSum;
        bestClusterAccum = cU32;
      }
    }
    unsigned int bestCluster = bestClusterAccum;
    assignments[gid] = bestCluster;
  }
}
