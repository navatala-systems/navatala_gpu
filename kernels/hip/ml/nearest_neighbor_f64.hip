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
extern "C" __global__ void navatala_ml_nearest_neighbor_f64(const double* D, const unsigned int* n, const unsigned int* m, unsigned int* indices, double* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int mVal = m[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    unsigned int rowBase = (rowIdx * mVal);
    double initDist = D[rowBase];
    double minDist = initDist;
    unsigned int minIdx = 0u;
    for (int j = 0; j < (int)(mVal); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int colIdx = (rowBase + jU32);
      double dist = D[colIdx];
      double currentMinDist = minDist;
      unsigned int currentMinIdx = minIdx;
      bool isSmaller = (dist < currentMinDist);
      double newMinDist = ((isSmaller) ? (dist) : (currentMinDist));
      unsigned int newMinIdx = ((isSmaller) ? (jU32) : (currentMinIdx));
      minDist = newMinDist;
      minIdx = newMinIdx;
    }
    unsigned int finalMinIdx = minIdx;
    double finalMinDist = minDist;
    indices[rowIdx] = finalMinIdx;
    distances[rowIdx] = finalMinDist;
  }
}
