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

kernel void navatala_cfd_primitives_sn_grad_scalar(device const float* cellVal [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* delta [[buffer(3)]], device const float* faceVal [[buffer(4)]], device const uint* faceMask [[buffer(5)]], device const int* params [[buffer(6)]], device float* outSnGrad [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float aO = cellVal[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float aN = cellVal[n];
      outSnGrad[((int)(int(__gid.x)))] = (delta[((int)(int(__gid.x)))] * (aN - aO));
    } else {
      if ((faceMask[((int)(int(__gid.x)))] != 0u)) {
        float aN = faceVal[((int)(int(__gid.x)))];
        outSnGrad[((int)(int(__gid.x)))] = (delta[((int)(int(__gid.x)))] * (aN - aO));
      } else {
        outSnGrad[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
      }
    }
  }
}
