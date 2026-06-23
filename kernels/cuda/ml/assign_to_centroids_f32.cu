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
extern "C" __global__ void navatala_ml_assign_to_centroids_f32(const float* data, const float* centroids, const unsigned int* n, const unsigned int* k, const unsigned int* d, unsigned int* assignments) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int dVal = d[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int dataBase = (gid * dVal);
    unsigned int bestClusterAccum = 0u;
    float bestDistAccum = __uint_as_float(0x7f7fffffu);
    float distSumAccum = __uint_as_float(0x00000000u);
    for (int c = 0; c < (int)(kVal); ++c) {
      unsigned int cU32 = ((unsigned int)(c));
      unsigned int centroidBase = (cU32 * dVal);
      distSumAccum = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(dVal); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        unsigned int dataIdx = (dataBase + jU32);
        unsigned int centIdx = (centroidBase + jU32);
        float dataVal = data[dataIdx];
        float centVal = centroids[centIdx];
        float diff = (dataVal - centVal);
        float diffSq = (diff * diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      float bestDist = bestDistAccum;
      if (distSum < bestDist) {
        bestDistAccum = distSum;
        bestClusterAccum = cU32;
      }
    }
    unsigned int bestCluster = bestClusterAccum;
    assignments[gid] = bestCluster;
  }
}
