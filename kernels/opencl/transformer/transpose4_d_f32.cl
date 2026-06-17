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

__kernel void navatala_transformer_transpose4_d_f32(__global const float* _input, __global const uint* dim0, __global const uint* dim1, __global const uint* dim2, __global const uint* dim3, __global const uint* perm0, __global const uint* perm1, __global const uint* perm2, __global const uint* perm3, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d0 = dim0[(uint)(0u)];
  uint d1 = dim1[(uint)(0u)];
  uint d2 = dim2[(uint)(0u)];
  uint d3 = dim3[(uint)(0u)];
  uint p0 = perm0[(uint)(0u)];
  uint p1 = perm1[(uint)(0u)];
  uint p2 = perm2[(uint)(0u)];
  uint p3 = perm3[(uint)(0u)];
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
    uint outD0 = (((p0 == (uint)(0u))) ? (d0) : ((((p0 == (uint)(1u))) ? (d1) : ((((p0 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD1 = (((p1 == (uint)(0u))) ? (d0) : ((((p1 == (uint)(1u))) ? (d1) : ((((p1 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD2 = (((p2 == (uint)(0u))) ? (d0) : ((((p2 == (uint)(1u))) ? (d1) : ((((p2 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD3 = (((p3 == (uint)(0u))) ? (d0) : ((((p3 == (uint)(1u))) ? (d1) : ((((p3 == (uint)(2u))) ? (d2) : (d3))))));
    uint outI0 = (((p0 == (uint)(0u))) ? (i0) : ((((p0 == (uint)(1u))) ? (i1) : ((((p0 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI1 = (((p1 == (uint)(0u))) ? (i0) : ((((p1 == (uint)(1u))) ? (i1) : ((((p1 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI2 = (((p2 == (uint)(0u))) ? (i0) : ((((p2 == (uint)(1u))) ? (i1) : ((((p2 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI3 = (((p3 == (uint)(0u))) ? (i0) : ((((p3 == (uint)(1u))) ? (i1) : ((((p3 == (uint)(2u))) ? (i2) : (i3))))));
    uint outStride1 = ((outD1 * outD2) * outD3);
    uint outStride2 = (outD2 * outD3);
    uint outStride3 = outD3;
    uint outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    float x = _input[gid];
    _output[outIdx] = x;
  }
}
