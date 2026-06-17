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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_compute_mutual_reachability_f64(__global const double* distances, __global const double* core_dists, __global const uint* n_samples, __global double* mutual_reach) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint totalPairs = (nSamples * nSamples);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / nSamples);
    uint j = (gid % nSamples);
    double dist = distances[gid];
    double coreI = core_dists[i];
    double coreJ = core_dists[j];
    bool coreIGtJ = (coreI > coreJ);
    double maxCore = as_double(0x0000000000000000ul);
    if (coreIGtJ) {
      maxCore = coreI;
    } else {
      maxCore = coreJ;
    }
    double maxCoreVal = maxCore;
    bool distGtCore = (dist > maxCoreVal);
    double mreach = as_double(0x0000000000000000ul);
    if (distGtCore) {
      mreach = dist;
    } else {
      mreach = maxCoreVal;
    }
    double finalMreach = mreach;
    mutual_reach[gid] = finalMreach;
  }
}
