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

__kernel void navatala_transformer_gelu_backward_f32(__global const float* _input, __global const float* gradOutput, __global const uint* count, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float u = (as_float(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (as_float(0x3f800000u) - tanh2);
    float threeCoef = as_float(0x3e095d4fu);
    float derivInner = (as_float(0x3f800000u) + (threeCoef * x2));
    float term2a = (as_float(0x3f000000u) * x);
    float term2b = (term2a * sech2);
    float term2c = (term2b * as_float(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (as_float(0x3f800000u) + tanhU);
    float term1 = (as_float(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dx = (dy * geluPrime);
    gradInput[gid] = dx;
  }
}
