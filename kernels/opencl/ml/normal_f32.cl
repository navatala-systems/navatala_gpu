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

__kernel void navatala_ml_normal_f32(__global const uint* count, __global const uint* seed, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if ((gid < countVal)) {
    uint ak0 = ((seedVal + (gid * (uint)(2654435769u))) + (uint)(0u));
    uint ak1 = ((ak0 ^ (ak0 >> (uint)(16u))) * (uint)(2246822519u));
    uint ak2 = ((ak1 ^ (ak1 >> (uint)(13u))) * (uint)(3266489917u));
    uint ak3 = (ak2 ^ (ak2 >> (uint)(16u)));
    float u1 = (((float)(ak3)) * as_float(0x2f800000u));
    uint bk0 = ((seedVal + (gid * (uint)(2654435769u))) + (uint)(2246822519u));
    uint bk1 = ((bk0 ^ (bk0 >> (uint)(16u))) * (uint)(2246822519u));
    uint bk2 = ((bk1 ^ (bk1 >> (uint)(13u))) * (uint)(3266489917u));
    uint bk3 = (bk2 ^ (bk2 >> (uint)(16u)));
    float u2 = (((float)(bk3)) * as_float(0x2f800000u));
    float u1e = (u1 + as_float(0x33d6bf95u));
    float r = sqrt((as_float(0xc0000000u) * log(u1e)));
    float z = (r * cos((as_float(0x40c90fdbu) * u2)));
    float outV = z;
    _output[gid] = outV;
  }
}
