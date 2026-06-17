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
__kernel void navatala_ml_aggregate_tree_outputs_f64(__global const double* leaf_values, __global const uint* n_samples, __global const uint* n_trees, __global double* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nTrees = n_trees[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nTrees);
    double sumOutput = as_double(0x0000000000000000ul);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      uint leafValIdx = (sampleBase + tU32);
      double leafVal = leaf_values[leafValIdx];
      double currSum = sumOutput;
      double newSum = (currSum + leafVal);
      sumOutput = newSum;
    }
    double finalSum = sumOutput;
    double nTreesF = ((double)(nTrees));
    double avgPred = (finalSum / nTreesF);
    predictions[gid] = avgPred;
  }
}
