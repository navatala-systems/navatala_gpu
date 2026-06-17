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

kernel void navatala_linalg_eig_select_top_k_f32(device const float* eigenvalues [[buffer(0)]], device const float* eigenvectors [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* n [[buffer(3)]], device float* topKValues [[buffer(4)]], device float* topKVectors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  uint nVal = n[0];
  uint totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint col = (gid / nVal);
    uint row = (gid % nVal);
    bool isFirstRow = (row == 0u);
    if (isFirstRow) {
      float eigenval = eigenvalues[col];
      topKValues[col] = eigenval;
    }
    uint srcIdx = ((col * nVal) + row);
    float vecElem = eigenvectors[srcIdx];
    topKVectors[srcIdx] = vecElem;
  }
}
