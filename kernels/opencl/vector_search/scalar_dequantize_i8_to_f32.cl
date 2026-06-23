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

__kernel void navatala_vector_search_scalar_dequantize_i8_to_f32(__global const char* _input, __global const uint* n_vectors, __global const uint* dim, __global const float* mins, __global const float* maxs, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if (gid < total) {
    uint dim_idx = (gid % d);
    char qval = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float qval_f32 = ((float)(qval));
    float offset = (qval_f32 + as_float(0x43000000u));
    float normalized = (offset / as_float(0x437f0000u));
    float range = (max_val - min_val);
    float dequant = ((normalized * range) + min_val);
    _output[gid] = dequant;
  }
}
