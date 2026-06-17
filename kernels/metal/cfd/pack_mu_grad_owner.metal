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

kernel void navatala_cfd_pack_mu_grad_owner(device const float* muCell [[buffer(0)]], device const float* gXX [[buffer(1)]], device const float* gXY [[buffer(2)]], device const float* gXZ [[buffer(3)]], device const float* gYX [[buffer(4)]], device const float* gYY [[buffer(5)]], device const float* gYZ [[buffer(6)]], device const float* gZX [[buffer(7)]], device const float* gZY [[buffer(8)]], device const float* gZZ [[buffer(9)]], device const int* owner [[buffer(10)]], device const int* procFaces [[buffer(11)]], device const int* counts [[buffer(12)]], device float* outSend [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(counts[0])))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int o = owner[f];
    int base = (int(__gid.x) * 10);
    outSend[base] = muCell[o];
    int b1 = (base + 1);
    outSend[b1] = gXX[o];
    int b2 = (base + 2);
    outSend[b2] = gXY[o];
    int b3 = (base + 3);
    outSend[b3] = gXZ[o];
    int b4 = (base + 4);
    outSend[b4] = gYX[o];
    int b5 = (base + 5);
    outSend[b5] = gYY[o];
    int b6 = (base + 6);
    outSend[b6] = gYZ[o];
    int b7 = (base + 7);
    outSend[b7] = gZX[o];
    int b8 = (base + 8);
    outSend[b8] = gZY[o];
    int b9 = (base + 9);
    outSend[b9] = gZZ[o];
  }
}
