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

__kernel void navatala_dataframe_extract_skewness_f32(__global const uint* n, __global const float* M2, __global const float* M3, __global float* skewness) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    uint nVal = n[0];
    float m2Val = M2[0];
    float m3Val = M3[0];
    float nf = ((float)(nVal));
    float sqrtN = sqrt(nf);
    float sqrtM2 = sqrt(m2Val);
    float m2Pow15 = (m2Val * sqrtM2);
    float denomSafe = (((m2Val == as_float(0x00000000u))) ? (as_float(0x3f800000u)) : (m2Pow15));
    float numerator = (sqrtN * m3Val);
    float skewRaw = (numerator / denomSafe);
    float skewResult = (((m2Val == as_float(0x00000000u))) ? (as_float(0x00000000u)) : (skewRaw));
    skewness[0] = skewResult;
  }
}
