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

kernel void navatala_sparse_sparse_cg_iter_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* p [[buffer(4)]], device const float* r [[buffer(5)]], device float* q [[buffer(6)]], device const float* rTr [[buffer(7)]], device const float* alpha [[buffer(8)]], device const uint* n [[buffer(9)]], device float* x_new [[buffer(10)]], device float* r_new [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if ((i < nVal)) {
    float alphaVal = alpha[0u];
    float xi = x[i];
    float pi = p[i];
    float ri = r[i];
    float qi = q[i];
    float alphaPi = (alphaVal * pi);
    float xNew = (xi + alphaPi);
    x_new[i] = xNew;
    float alphaQi = (alphaVal * qi);
    float rNew = (ri - alphaQi);
    r_new[i] = rNew;
  }
}
