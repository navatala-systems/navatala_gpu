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
extern "C" __global__ void navatala_ml_compute_core_distances_f32(const float* distances, const unsigned int* n_samples, const unsigned int* min_samples, float* core_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int minSamples = min_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int rowBase = (gid * nSamples);
    float kthDist = __uint_as_float(0x7149f2cau);
    unsigned int countSmaller = 0u;
    for (int iter = 0; iter < (int)(nSamples); ++iter) {
      unsigned int iterU32 = ((unsigned int)(iter));
      unsigned int distIdx = (rowBase + iterU32);
      float candidateDist = distances[distIdx];
      bool notSelf = (candidateDist > __uint_as_float(0x00000000u));
      if (notSelf) {
        countSmaller = 0u;
        for (int j = 0; j < (int)(nSamples); ++j) {
          unsigned int jU32 = ((unsigned int)(j));
          unsigned int jDistIdx = (rowBase + jU32);
          float jDist = distances[jDistIdx];
          bool isPositive = (jDist > __uint_as_float(0x00000000u));
          bool isLeCandidate = (jDist <= candidateDist);
          bool shouldCount = (isPositive && isLeCandidate);
          if (shouldCount) {
            unsigned int currCount = countSmaller;
            unsigned int newCount = (currCount + 1u);
            countSmaller = newCount;
          }
        }
        unsigned int finalCount = countSmaller;
        bool isKthOrMore = (finalCount >= minSamples);
        if (isKthOrMore) {
          float currKth = kthDist;
          bool isBetter = (candidateDist < currKth);
          if (isBetter) {
            kthDist = candidateDist;
          }
        }
      }
    }
    float finalCoreDist = kthDist;
    core_dists[gid] = finalCoreDist;
  }
}
