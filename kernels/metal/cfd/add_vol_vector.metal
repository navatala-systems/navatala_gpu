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

kernel void navatala_cfd_add_vol_vector(device const float* ax [[buffer(0)]], device const float* ay [[buffer(1)]], device const float* az [[buffer(2)]], device const float* bx [[buffer(3)]], device const float* by_ [[buffer(4)]], device const float* bz [[buffer(5)]], device const int* params [[buffer(6)]], device float* outX [[buffer(7)]], device float* outY [[buffer(8)]], device float* outZ [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    outX[((int)(int(__gid.x)))] = (ax[((int)(int(__gid.x)))] + bx[((int)(int(__gid.x)))]);
    outY[((int)(int(__gid.x)))] = (ay[((int)(int(__gid.x)))] + by_[((int)(int(__gid.x)))]);
    outZ[((int)(int(__gid.x)))] = (az[((int)(int(__gid.x)))] + bz[((int)(int(__gid.x)))]);
  }
}
