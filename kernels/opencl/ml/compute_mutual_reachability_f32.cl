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

__kernel void navatala_ml_compute_mutual_reachability_f32(__global const float* distances, __global const float* core_dists, __global const uint* n_samples, __global float* mutual_reach) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint totalPairs = (nSamples * nSamples);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / nSamples);
    uint j = (gid % nSamples);
    float dist = distances[gid];
    float coreI = core_dists[i];
    float coreJ = core_dists[j];
    bool coreIGtJ = (coreI > coreJ);
    float maxCore = as_float(0x00000000u);
    if (coreIGtJ) {
      maxCore = coreI;
    } else {
      maxCore = coreJ;
    }
    float maxCoreVal = maxCore;
    bool distGtCore = (dist > maxCoreVal);
    float mreach = as_float(0x00000000u);
    if (distGtCore) {
      mreach = dist;
    } else {
      mreach = maxCoreVal;
    }
    float finalMreach = mreach;
    mutual_reach[gid] = finalMreach;
  }
}
