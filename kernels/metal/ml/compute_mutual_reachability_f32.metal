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

kernel void navatala_ml_compute_mutual_reachability_f32(device const float* distances [[buffer(0)]], device const float* core_dists [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device float* mutual_reach [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
    float maxCore = as_type<float>(0x00000000u);
    if (coreIGtJ) {
      maxCore = coreI;
    } else {
      maxCore = coreJ;
    }
    float maxCoreVal = maxCore;
    bool distGtCore = (dist > maxCoreVal);
    float mreach = as_type<float>(0x00000000u);
    if (distGtCore) {
      mreach = dist;
    } else {
      mreach = maxCoreVal;
    }
    float finalMreach = mreach;
    mutual_reach[gid] = finalMreach;
  }
}
