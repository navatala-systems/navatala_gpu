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

kernel void navatala_dataframe_welford_update_f32(device const float* x [[buffer(0)]], device uint* count [[buffer(1)]], device float* mean [[buffer(2)]], device float* M2 [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    uint nOld = count[0];
    float meanOld = mean[0];
    float m2Old = M2[0];
    float xVal = x[0];
    uint nNew = (nOld + 1u);
    float nNewF = ((float)(nNew));
    float delta = (xVal - meanOld);
    float meanNew = (meanOld + (delta / nNewF));
    float delta2 = (xVal - meanNew);
    float m2New = (m2Old + (delta * delta2));
    count[0] = nNew;
    mean[0] = meanNew;
    M2[0] = m2New;
  }
}
