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

kernel void navatala_cfd_dot_partials(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outPartials [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float tmp[256];
  float v = as_type<float>(0x00000000u);
  if (((int)(int(__gid.x))) < counts[0]) {
    v = (a[((int)(int(__gid.x)))] * b[((int)(int(__gid.x)))]);
  }
  tmp[((int)(int(__tid.x)))] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 128) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 128)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 64) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 64)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 32) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 32)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 16) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 16)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 8) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 8)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 4) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 4)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 2) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 2)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 1) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 1)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) == 0) {
    outPartials[((int)(int(__tgid.x)))] = tmp[0];
  }
}
