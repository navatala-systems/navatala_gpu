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

__kernel void navatala_transformer_geglu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float gate2 = (gate * gate);
    float gate3 = (gate2 * gate);
    float cubeTerm = (as_float(0x3d372713u) * gate3);
    float inner = (gate + cubeTerm);
    float tanhArg = (as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_float(0x3f800000u) + tanhVal);
    float halfGate = (as_float(0x3f000000u) * gate);
    float geluGate = (halfGate * onePlusTanh);
    float result = (geluGate * value);
    _output[gid] = result;
  }
}
