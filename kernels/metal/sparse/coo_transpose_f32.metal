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

kernel void navatala_sparse_coo_transpose_f32(device const uint* rowInd [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nnz [[buffer(3)]], device uint* rowInd_out [[buffer(4)]], device uint* colInd_out [[buffer(5)]], device float* values_out [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  if ((i < nnz[0u])) {
    uint col = colInd[i];
    rowInd_out[i] = col;
    uint row = rowInd[i];
    colInd_out[i] = row;
    float val = values[i];
    values_out[i] = val;
  }
}
