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

__kernel void navatala_transformer_cast_f32_to_f_p8_e4_m3(__global const float* _input, __global const float* scale, __global const uint* count, __global uchar* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[(uint)(0u)];
    float scaled = (x * s);
    float minVal = as_float(0xc3700000u);
    float maxVal = as_float(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    _output[gid] = fp8Val;
  }
}
