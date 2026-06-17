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
extern "C" __global__ void navatala_ml_compute_mutual_reachability_f32(const float* distances, const float* core_dists, const unsigned int* n_samples, float* mutual_reach) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int totalPairs = (nSamples * nSamples);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int i = (gid / nSamples);
    unsigned int j = (gid % nSamples);
    float dist = distances[gid];
    float coreI = core_dists[i];
    float coreJ = core_dists[j];
    bool coreIGtJ = (coreI > coreJ);
    float maxCore = __uint_as_float(0x00000000u);
    if (coreIGtJ) {
      maxCore = coreI;
    } else {
      maxCore = coreJ;
    }
    float maxCoreVal = maxCore;
    bool distGtCore = (dist > maxCoreVal);
    float mreach = __uint_as_float(0x00000000u);
    if (distGtCore) {
      mreach = dist;
    } else {
      mreach = maxCoreVal;
    }
    float finalMreach = mreach;
    mutual_reach[gid] = finalMreach;
  }
}
