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

__kernel void navatala_ml_avgpool_exclude_pad_f32(__global const float* _input, __global const uint* window, __global const uint* stride, __global const uint* inCount, __global const uint* outCount, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float sum = as_float(0x00000000u);
    float cnt = as_float(0x00000000u);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = _input[idx];
        sum = (sum + v);
        cnt = (cnt + as_float(0x3f800000u));
      }
    }
    _output[gid] = (sum / (((cnt > as_float(0x00000000u))) ? (cnt) : (as_float(0x3f800000u))));
  }
}
