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

kernel void navatala_ml_compute_leaf_class_probs_f32(device const int* labels [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_leaf_samples [[buffer(2)]], device const uint* n_classes [[buffer(3)]], device float* probs [[buffer(4)]], device uint* counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_leaf_samples[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    uint labelU32 = ((uint)(label));
    uint currCount = counts[labelU32];
    uint newCount = (currCount + 1u);
    counts[labelU32] = newCount;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool inBoundsClass = (gid < nClasses);
  if (inBoundsClass) {
    uint count = counts[gid];
    float countF = ((float)(count));
    float nSamplesF = ((float)(nSamples));
    float prob = (countF / nSamplesF);
    probs[gid] = prob;
  }
}
