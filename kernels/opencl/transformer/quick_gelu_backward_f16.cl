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
__kernel void navatala_transformer_quick_gelu_backward_f16(__global const half* _input, __global const half* gradOutput, __global const uint* count, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float scaledX = (as_float(0x3fd9db23u) * xF32);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_float(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_float(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dxF32 = (dyF32 * quickGeluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}
