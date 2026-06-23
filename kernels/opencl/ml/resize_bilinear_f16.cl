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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_resize_bilinear_f16(__global const half* _input, __global const uint* inSize, __global const uint* outSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    uint inSizeVal = inSize[0];
    uint inM1 = (inSizeVal - (uint)(1u));
    uint outM1 = (outSizeVal - (uint)(1u));
    float pos = ((((float)(gid)) * ((float)(inM1))) / ((float)(outM1)));
    float i0f = floor(pos);
    uint i0 = ((uint)(i0f));
    float frac = (pos - i0f);
    uint i0p1 = (i0 + (uint)(1u));
    uint i1 = (((i0p1 < inM1)) ? (i0p1) : (inM1));
    float v0 = ((float)(_input[i0]));
    float v1 = ((float)(_input[i1]));
    float outF = ((v0 * (as_float(0x3f800000u) - frac)) + (v1 * frac));
    _output[gid] = ((half)(outF));
  }
}
