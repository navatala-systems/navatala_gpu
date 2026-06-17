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

kernel void navatala_dataframe_center_matrix_f32(device const float* _input [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device const float* means [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint colIdx = (gid % d);
    float x = _input[gid];
    float colMean = means[colIdx];
    float centered = (x - colMean);
    _output[gid] = centered;
  }
}
