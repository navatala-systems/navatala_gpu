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
__kernel void navatala_transformer_dropout_backward_f16(__global const half* gradOutput, __global const uint* count, __global const float* prob, __global const uint* seed, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  float p = prob[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half dyF16 = gradOutput[gid];
    float dy = ((float)(dyF16));
    uint h0 = (gid ^ s);
    uint h1 = (h0 * (uint)(3432918353u));
    uint h2 = (h1 ^ (h1 >> (uint)(15u)));
    uint h3 = (h2 * (uint)(461845907u));
    uint hashVal = (h3 ^ (h3 >> (uint)(13u)));
    float randF32 = (((float)(hashVal)) / as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_float(0x3f800000u) - p);
    float scale = (as_float(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dxF32 = ((keep) ? (scaled) : (as_float(0x00000000u)));
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}
