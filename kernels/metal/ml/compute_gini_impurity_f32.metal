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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_gini_impurity_f32(device const int* labels [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_node_samples [[buffer(2)]], device const uint* n_classes [[buffer(3)]], device float* gini [[buffer(4)]], device uint* class_counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint nSamples = n_node_samples[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    uint labelU32 = ((uint)(label));
    uint currCount = class_counts[labelU32];
    uint newCount = (currCount + 1u);
    class_counts[labelU32] = newCount;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool isFirstThread = (gid == 0u);
  if (isFirstThread) {
    float sumPSquared = as_type<float>(0x00000000u);
    float nSamplesF = ((float)(nSamples));
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint count = class_counts[cU32];
      float countF = ((float)(count));
      float p = (countF / nSamplesF);
      float pSquared = (p * p);
      float currSum = sumPSquared;
      float newSum = (currSum + pSquared);
      sumPSquared = newSum;
    }
    float finalSum = sumPSquared;
    float giniVal = (as_type<float>(0x3f800000u) - finalSum);
    gini[0] = giniVal;
  }
}
