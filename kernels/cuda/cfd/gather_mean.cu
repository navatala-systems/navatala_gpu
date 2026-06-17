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
extern "C" __global__ void navatala_cfd_gather_mean(const float* features, const int* neighbors, const int* neighborCounts, const int* numNodes, const int* featureDim, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nNodes = numNodes[0];
  if ((gid < nNodes)) {
    int cnt = neighborCounts[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      float acc = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(cnt); ++j) {
        int nbrFlat = ((gid * 8) + j);
        int nbr = neighbors[nbrFlat];
        int featFlat = ((nbr * fd) + f);
        float v = features[featFlat];
        acc = (acc + v);
      }
      float cntF = ((float)(cnt));
      float mean = (((cnt > 0)) ? ((acc / cntF)) : (__uint_as_float(0x00000000u)));
      int outFlat = ((gid * fd) + f);
      _out[outFlat] = mean;
    }
  }
}
