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
extern "C" __global__ void navatala_ml_compute_mutual_reachability_f64(const double* distances, const double* core_dists, const unsigned int* n_samples, double* mutual_reach) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int totalPairs = (nSamples * nSamples);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int i = (gid / nSamples);
    unsigned int j = (gid % nSamples);
    double dist = distances[gid];
    double coreI = core_dists[i];
    double coreJ = core_dists[j];
    bool coreIGtJ = (coreI > coreJ);
    double maxCore = __longlong_as_double(0x0000000000000000ull);
    if (coreIGtJ) {
      maxCore = coreI;
    } else {
      maxCore = coreJ;
    }
    double maxCoreVal = maxCore;
    bool distGtCore = (dist > maxCoreVal);
    double mreach = __longlong_as_double(0x0000000000000000ull);
    if (distGtCore) {
      mreach = dist;
    } else {
      mreach = maxCoreVal;
    }
    double finalMreach = mreach;
    mutual_reach[gid] = finalMreach;
  }
}
