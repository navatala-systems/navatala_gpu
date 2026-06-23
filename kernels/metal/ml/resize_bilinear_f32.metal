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

kernel void navatala_ml_resize_bilinear_f32(device const float* _input [[buffer(0)]], device const uint* inSize [[buffer(1)]], device const uint* outSize [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    uint inSizeVal = inSize[0];
    uint inM1 = (inSizeVal - 1u);
    uint outM1 = (outSizeVal - 1u);
    float pos = ((((float)(gid)) * ((float)(inM1))) / ((float)(outM1)));
    float i0f = floor(pos);
    uint i0 = ((uint)(i0f));
    float frac = (pos - i0f);
    uint i0p1 = (i0 + 1u);
    uint i1 = (((i0p1 < inM1)) ? (i0p1) : (inM1));
    float v0 = _input[i0];
    float v1 = _input[i1];
    float outF = ((v0 * (as_type<float>(0x3f800000u) - frac)) + (v1 * frac));
    _output[gid] = outF;
  }
}
