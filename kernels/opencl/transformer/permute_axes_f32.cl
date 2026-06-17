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

__kernel void navatala_transformer_permute_axes_f32(__global const float* _input, __global const uint* dim0, __global const uint* dim1, __global const uint* dim2, __global const uint* dim3, __global const uint* perm0, __global const uint* perm1, __global const uint* perm2, __global const uint* perm3, __global float* _output) {
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
  uint total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride0 = (d1 * (d2 * d3));
    uint stride1 = (d2 * d3);
    uint stride2 = d3;
    uint stride3 = (uint)(1u);
    uint i0 = (gid / stride0);
    uint rem0 = (gid % stride0);
    uint i1 = (rem0 / stride1);
    uint rem1 = (rem0 % stride1);
    uint i2 = (rem1 / stride2);
    uint i3 = (rem1 % stride2);
    float val = _input[gid];
    uint od0 = (((p0 == (uint)(0u))) ? (d0) : ((((p0 == (uint)(1u))) ? (d1) : ((((p0 == (uint)(2u))) ? (d2) : (d3))))));
    uint od1 = (((p1 == (uint)(0u))) ? (d0) : ((((p1 == (uint)(1u))) ? (d1) : ((((p1 == (uint)(2u))) ? (d2) : (d3))))));
    uint od2 = (((p2 == (uint)(0u))) ? (d0) : ((((p2 == (uint)(1u))) ? (d1) : ((((p2 == (uint)(2u))) ? (d2) : (d3))))));
    uint od3 = (((p3 == (uint)(0u))) ? (d0) : ((((p3 == (uint)(1u))) ? (d1) : ((((p3 == (uint)(2u))) ? (d2) : (d3))))));
    uint oi0 = (((p0 == (uint)(0u))) ? (i0) : ((((p0 == (uint)(1u))) ? (i1) : ((((p0 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi1 = (((p1 == (uint)(0u))) ? (i0) : ((((p1 == (uint)(1u))) ? (i1) : ((((p1 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi2 = (((p2 == (uint)(0u))) ? (i0) : ((((p2 == (uint)(1u))) ? (i1) : ((((p2 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi3 = (((p3 == (uint)(0u))) ? (i0) : ((((p3 == (uint)(1u))) ? (i1) : ((((p3 == (uint)(2u))) ? (i2) : (i3))))));
    uint ostride0 = (od1 * (od2 * od3));
    uint ostride1 = (od2 * od3);
    uint ostride2 = od3;
    uint outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}
