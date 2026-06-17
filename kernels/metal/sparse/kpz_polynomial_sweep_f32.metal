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

kernel void navatala_sparse_kpz_polynomial_sweep_f32(device const float* Ax [[buffer(0)]], device const float* x [[buffer(1)]], device const float* b [[buffer(2)]], device const float* kpzCoeffs [[buffer(3)]], device const uint* degree [[buffer(4)]], device const uint* n [[buffer(5)]], device float* xNew [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    float ri = (b[i] - Ax[i]);
    float xi = x[i];
    int deg = ((int)(degree[0]));
    float poly = as_type<float>(0x00000000u);
    float power = as_type<float>(0x3f800000u);
    for (int k = 0; k < (int)(deg); ++k) {
      float ck = kpzCoeffs[k];
      poly = (poly + (ck * power));
      power = (power * ri);
    }
    xNew[i] = (xi + poly);
  }
}
