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
extern "C" __global__ void navatala_graph_construct_indicator_f32(const unsigned int* clusters, const unsigned int* clusterIndex, const unsigned int* n, float* indicator) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    float indicatorVal = ((isTarget) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    indicator[gid] = indicatorVal;
  }
}
