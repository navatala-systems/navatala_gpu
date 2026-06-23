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
__kernel void navatala_ml_maxpool_f16(__global const half* _input, __global const uint* window, __global const uint* stride, __global const uint* inCount, __global const uint* outCount, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float acc = as_float(0xff7fc99eu);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = ((float)(_input[idx]));
        acc = (((acc > v)) ? (acc) : (v));
      }
    }
    _output[gid] = ((half)(acc));
  }
}
