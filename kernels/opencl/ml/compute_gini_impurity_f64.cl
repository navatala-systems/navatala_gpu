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
__kernel void navatala_ml_compute_gini_impurity_f64(__global const int* labels, __global const uint* sample_indices, __global const uint* n_node_samples, __global const uint* n_classes, __global double* gini, __global uint* class_counts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_node_samples[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    uint labelU32 = ((uint)(label));
    uint currCount = class_counts[labelU32];
    uint newCount = (currCount + (uint)(1u));
    class_counts[labelU32] = newCount;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool isFirstThread = (gid == (uint)(0u));
  if (isFirstThread) {
    double sumPSquared = as_double(0x0000000000000000ul);
    double nSamplesF = ((double)(nSamples));
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint count = class_counts[cU32];
      double countF = ((double)(count));
      double p = (countF / nSamplesF);
      double pSquared = (p * p);
      double currSum = sumPSquared;
      double newSum = (currSum + pSquared);
      sumPSquared = newSum;
    }
    double finalSum = sumPSquared;
    double giniVal = (as_double(0x3ff0000000000000ul) - finalSum);
    gini[0] = giniVal;
  }
}
