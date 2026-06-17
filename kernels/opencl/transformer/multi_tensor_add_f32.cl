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

__kernel void navatala_transformer_multi_tensor_add_f32(__global const float* _input, __global const float* value, __global const uint* totalElements, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float v = value[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x + v);
    _output[gid] = result;
  }
}
