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
__kernel void navatala_transformer_gelu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half x2 = (x * x);
    half x3 = (x2 * x);
    half cubeTerm = ((half)(0.044715f) * x3);
    half inner = (x + cubeTerm);
    half tanhArg = ((half)(0.797885f) * inner);
    half tanhVal = tanh(tanhArg);
    half onePlusTanh = ((half)(1.000000f) + tanhVal);
    half halfX = ((half)(0.500000f) * x);
    half result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}
