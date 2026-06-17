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

kernel void navatala_transformer_transpose4_d_f16(device const half* _input [[buffer(0)]], device const uint* dim0 [[buffer(1)]], device const uint* dim1 [[buffer(2)]], device const uint* dim2 [[buffer(3)]], device const uint* dim3 [[buffer(4)]], device const uint* perm0 [[buffer(5)]], device const uint* perm1 [[buffer(6)]], device const uint* perm2 [[buffer(7)]], device const uint* perm3 [[buffer(8)]], device half* _output [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d0 = dim0[0u];
  uint d1 = dim1[0u];
  uint d2 = dim2[0u];
  uint d3 = dim3[0u];
  uint p0 = perm0[0u];
  uint p1 = perm1[0u];
  uint p2 = perm2[0u];
  uint p3 = perm3[0u];
  uint total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride1 = ((d1 * d2) * d3);
    uint stride2 = (d2 * d3);
    uint stride3 = d3;
    uint i0 = (gid / stride1);
    uint rem0 = (gid % stride1);
    uint i1 = (rem0 / stride2);
    uint rem1 = (rem0 % stride2);
    uint i2 = (rem1 / stride3);
    uint i3 = (rem1 % stride3);
    uint outD0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    uint outD1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    uint outD2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    uint outD3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    uint outI0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    uint outI1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    uint outI2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    uint outI3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    uint outStride1 = ((outD1 * outD2) * outD3);
    uint outStride2 = (outD2 * outD3);
    uint outStride3 = outD3;
    uint outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    half x = _input[gid];
    _output[outIdx] = x;
  }
}
