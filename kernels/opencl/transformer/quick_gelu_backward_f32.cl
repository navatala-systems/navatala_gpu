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

__kernel void navatala_transformer_quick_gelu_backward_f32(__global const float* _input, __global const float* gradOutput, __global const uint* count, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float scaledX = (as_float(0x3fd9db23u) * x);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_float(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_float(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dx = (dy * quickGeluPrime);
    gradInput[gid] = dx;
  }
}
