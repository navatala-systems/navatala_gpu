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

__kernel void navatala_vector_search_scalar_quantize_f32_to_i8(__global const float* _input, __global const uint* n_vectors, __global const uint* dim, __global const float* mins, __global const float* maxs, __global char* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if (gid < total) {
    uint dim_idx = (gid % d);
    float val = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float range = (max_val - min_val);
    float normalized = ((val - min_val) / range);
    float scaled = (normalized * as_float(0x437f0000u));
    float offset = (scaled - as_float(0x43000000u));
    float neg128 = (as_float(0x00000000u) - as_float(0x43000000u));
    float clamped = (((offset < neg128)) ? (neg128) : ((((offset > as_float(0x42fe0000u))) ? (as_float(0x42fe0000u)) : (offset))));
    int quantized_i32 = ((int)(clamped));
    char quantized = ((char)(quantized_i32));
    _output[gid] = quantized;
  }
}
