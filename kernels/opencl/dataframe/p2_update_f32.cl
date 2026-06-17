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

__kernel void navatala_dataframe_p2_update_f32(__global const float* x, __global float* q0, __global float* q1, __global float* q2, __global float* q3, __global float* q4, __global uint* pos0, __global uint* pos1, __global uint* pos2, __global uint* pos3, __global uint* pos4, __global const float* targetP, __global uint* count) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    float xVal = x[0];
    uint n = count[0];
    float pVal = targetP[0];
    float h0 = q0[0];
    float h1 = q1[0];
    float h2 = q2[0];
    float h3 = q3[0];
    float h4 = q4[0];
    float newH0 = (((n == (uint)(0u))) ? (xVal) : ((((xVal < h0)) ? (xVal) : (h0))));
    float finalH0 = (((n >= (uint)(5u))) ? ((((xVal < h0)) ? (xVal) : (h0))) : (newH0));
    float finalH4 = (((n >= (uint)(5u))) ? ((((xVal > h4)) ? (xVal) : (h4))) : ((((n == (uint)(4u))) ? (xVal) : (h4))));
    float alpha = as_float(0x3c23d70au);
    float oneMinusAlpha = (as_float(0x3f800000u) - alpha);
    float finalH2 = (((n >= (uint)(5u))) ? (((oneMinusAlpha * h2) + (alpha * xVal))) : ((((n == (uint)(2u))) ? (xVal) : (h2))));
    q0[0] = finalH0;
    q2[0] = finalH2;
    q4[0] = finalH4;
    uint newCount = (n + (uint)(1u));
    count[0] = newCount;
  }
}
