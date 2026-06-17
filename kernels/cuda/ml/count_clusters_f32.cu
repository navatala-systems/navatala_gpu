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
extern "C" __global__ void navatala_ml_count_clusters_f32(const int* labels, const unsigned int* nPoints, int* numClusters) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ int sdata[256];
  unsigned int n = nPoints[0];
  bool inBounds = (gid < n);
  int myLabel = ((inBounds) ? (labels[gid]) : (-1));
  sdata[lid] = myLabel;
  __syncthreads();
  unsigned int countClstF32RedStride = 128u;
  for (int countClstF32RedStep = 0; countClstF32RedStep < (int)(8); ++countClstF32RedStep) {
    unsigned int countClstF32Stride = countClstF32RedStride;
    if ((lid < countClstF32Stride)) {
      unsigned int otherIdx = (lid + countClstF32Stride);
      int other = sdata[otherIdx];
      int mine = sdata[lid];
      bool isGreater = (other > mine);
      int maxVal = ((isGreater) ? (other) : (mine));
      sdata[lid] = maxVal;
    }
    unsigned int countClstF32StrideToHalve = countClstF32RedStride;
    unsigned int countClstF32NextStride = (countClstF32StrideToHalve >> 1u);
    countClstF32RedStride = countClstF32NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    int maxLabel = sdata[0];
    int count = (maxLabel + 1);
    numClusters[0] = count;
  }
}
