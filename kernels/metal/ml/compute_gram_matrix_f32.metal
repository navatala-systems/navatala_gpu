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

kernel void navatala_ml_compute_gram_matrix_f32(device const float* X [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device float* G [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint p = nFeatures[0];
  uint totalElems = (p * p);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / p);
    uint j = (gid % p);
    float dotProduct = as_type<float>(0x00000000u);
    uint idx0_i = i;
    uint idx0_j = j;
    float x0_i = X[idx0_i];
    float x0_j = X[idx0_j];
    float prod0 = (x0_i * x0_j);
    float sum = prod0;
    G[gid] = sum;
  }
}
