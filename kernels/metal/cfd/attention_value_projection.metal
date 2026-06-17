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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_attention_value_projection(device const float* weights [[buffer(0)]], device const float* v [[buffer(1)]], device const int* numQ [[buffer(2)]], device const int* numK [[buffer(3)]], device const int* headDim [[buffer(4)]], device float* _out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * hd);
  if ((gid < total)) {
    int qi = (gid / hd);
    int d = (gid % hd);
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int wIdx = ((qi * nk) + j);
      int vIdx = ((j * hd) + d);
      float w = weights[wIdx];
      float vv = v[vIdx];
      acc = (acc + (w * vv));
    }
    _out[gid] = acc;
  }
}
