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

kernel void navatala_sparse_chebyshev_coeff_update_f32(device const float* theta [[buffer(0)]], device const float* delta [[buffer(1)]], device const float* sigma1 [[buffer(2)]], device const float* sigmaOld [[buffer(3)]], device float* coeffA [[buffer(4)]], device float* coeffB [[buffer(5)]], device float* sigmaNew [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float th = theta[0];
  float de = delta[0];
  float s1 = sigma1[0];
  float so = sigmaOld[0];
  float twoThOverDe = ((as_type<float>(0x40000000u) * th) / de);
  float sn = (as_type<float>(0x3f800000u) / (twoThOverDe - so));
  float cA = ((as_type<float>(0x40000000u) * sn) / (de * s1));
  float cB = (sn * (so - as_type<float>(0x3f800000u)));
  coeffA[0] = cA;
  coeffB[0] = cB;
  sigmaNew[0] = sn;
}
