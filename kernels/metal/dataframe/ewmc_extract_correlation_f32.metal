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

kernel void navatala_dataframe_ewmc_extract_correlation_f32(device const float* Cxy [[buffer(0)]], device const float* varX [[buffer(1)]], device const float* varY [[buffer(2)]], device const float* nEff [[buffer(3)]], device float* result [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float c = Cxy[0];
    float vx = varX[0];
    float vy = varY[0];
    float n = nEff[0];
    float vxvy = (vx * vy);
    float eps = as_type<float>(0x2edbe6ffu);
    float denom = sqrt(vxvy);
    float safeDenom = (((vxvy < eps)) ? (as_type<float>(0x3f800000u)) : (denom));
    float corr = (c / safeDenom);
    float res = (((vxvy < eps)) ? (as_type<float>(0x00000000u)) : (corr));
    result[0] = res;
  }
}
