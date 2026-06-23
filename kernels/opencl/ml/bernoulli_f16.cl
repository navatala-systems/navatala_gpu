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
__kernel void navatala_ml_bernoulli_f16(__global const uint* count, __global const uint* seed, __global const float* p, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint uk0 = ((seedVal + (gid * (uint)(2654435769u))) + (uint)(0u));
    uint uk1 = ((uk0 ^ (uk0 >> (uint)(16u))) * (uint)(2246822519u));
    uint uk2 = ((uk1 ^ (uk1 >> (uint)(13u))) * (uint)(3266489917u));
    uint uk3 = (uk2 ^ (uk2 >> (uint)(16u)));
    float uF = (((float)(uk3)) * as_float(0x2f800000u));
    float pVal = p[0];
    float bit = (((uF < pVal)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half outV = ((half)(bit));
    _output[gid] = outV;
  }
}
