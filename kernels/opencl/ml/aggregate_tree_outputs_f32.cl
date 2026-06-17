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

__kernel void navatala_ml_aggregate_tree_outputs_f32(__global const float* leaf_values, __global const uint* n_samples, __global const uint* n_trees, __global float* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nTrees = n_trees[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nTrees);
    float sumOutput = as_float(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      uint leafValIdx = (sampleBase + tU32);
      float leafVal = leaf_values[leafValIdx];
      float currSum = sumOutput;
      float newSum = (currSum + leafVal);
      sumOutput = newSum;
    }
    float finalSum = sumOutput;
    float nTreesF = ((float)(nTrees));
    float avgPred = (finalSum / nTreesF);
    predictions[gid] = avgPred;
  }
}
