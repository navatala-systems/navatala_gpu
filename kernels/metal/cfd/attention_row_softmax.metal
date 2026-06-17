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

kernel void navatala_cfd_attention_row_softmax(device const float* scores [[buffer(0)]], device const int* numQ [[buffer(1)]], device const int* numK [[buffer(2)]], device float* weights [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int qi = int(__gid.x);
  int nq = numQ[0];
  int nk = numK[0];
  if ((qi < nq)) {
    int rowBase = (qi * nk);
    float maxScore = as_type<float>(0xf149f2cau);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      maxScore = (((s > maxScore)) ? (s) : (maxScore));
    }
    float sumExp = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      sumExp = (sumExp + e);
    }
    float denom = (sumExp + as_type<float>(0x3089705fu));
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      float w = (e / denom);
      weights[idx] = w;
    }
  }
}
