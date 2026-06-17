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
__kernel void navatala_ml_compute_leaf_class_probs_f64(__global const int* labels, __global const uint* sample_indices, __global const uint* n_leaf_samples, __global const uint* n_classes, __global double* probs, __global uint* counts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_leaf_samples[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    uint labelU32 = ((uint)(label));
    uint currCount = counts[labelU32];
    uint newCount = (currCount + (uint)(1u));
    counts[labelU32] = newCount;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool inBoundsClass = (gid < nClasses);
  if (inBoundsClass) {
    uint count = counts[gid];
    double countF = ((double)(count));
    double nSamplesF = ((double)(nSamples));
    double prob = (countF / nSamplesF);
    probs[gid] = prob;
  }
}
