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
__kernel void navatala_ml_compute_core_distances_f64(__global const double* distances, __global const uint* n_samples, __global const uint* min_samples, __global double* core_dists) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint minSamples = min_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint rowBase = (gid * nSamples);
    double kthDist = as_double(0x46293e5939a08ceaul);
    uint countSmaller = (uint)(0u);
    for (int iter = 0; iter < (int)(nSamples); ++iter) {
      uint iterU32 = ((uint)(iter));
      uint distIdx = (rowBase + iterU32);
      double candidateDist = distances[distIdx];
      bool notSelf = (candidateDist > as_double(0x0000000000000000ul));
      if (notSelf) {
        countSmaller = (uint)(0u);
        for (int j = 0; j < (int)(nSamples); ++j) {
          uint jU32 = ((uint)(j));
          uint jDistIdx = (rowBase + jU32);
          double jDist = distances[jDistIdx];
          bool isPositive = (jDist > as_double(0x0000000000000000ul));
          bool isLeCandidate = (jDist <= candidateDist);
          bool shouldCount = (isPositive && isLeCandidate);
          if (shouldCount) {
            uint currCount = countSmaller;
            uint newCount = (currCount + (uint)(1u));
            countSmaller = newCount;
          }
        }
        uint finalCount = countSmaller;
        bool isKthOrMore = (finalCount >= minSamples);
        if (isKthOrMore) {
          double currKth = kthDist;
          bool isBetter = (candidateDist < currKth);
          if (isBetter) {
            kthDist = candidateDist;
          }
        }
      }
    }
    double finalCoreDist = kthDist;
    core_dists[gid] = finalCoreDist;
  }
}
