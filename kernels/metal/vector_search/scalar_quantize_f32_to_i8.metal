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

kernel void navatala_vector_search_scalar_quantize_f32_to_i8(device const float* _input [[buffer(0)]], device const uint* n_vectors [[buffer(1)]], device const uint* dim [[buffer(2)]], device const float* mins [[buffer(3)]], device const float* maxs [[buffer(4)]], device char* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if ((gid < total)) {
    uint dim_idx = (gid % d);
    float val = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float range = (max_val - min_val);
    float normalized = ((val - min_val) / range);
    float scaled = (normalized * as_type<float>(0x437f0000u));
    float offset = (scaled - as_type<float>(0x43000000u));
    float neg128 = (as_type<float>(0x00000000u) - as_type<float>(0x43000000u));
    float clamped = (((offset < neg128)) ? (neg128) : ((((offset > as_type<float>(0x42fe0000u))) ? (as_type<float>(0x42fe0000u)) : (offset))));
    int quantized_i32 = ((int)(clamped));
    char quantized = ((char)(quantized_i32));
    _output[gid] = quantized;
  }
}
