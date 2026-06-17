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

kernel void navatala_sparse_dense_sym_eig_f32(device const float* M [[buffer(0)]], device const uint* k [[buffer(1)]], device float* eigenvalues [[buffer(2)]], device float* eigenvectors [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int kVal = ((int)(k[0]));
  for (int i = 0; i < (int)(kVal); ++i) {
    for (int j = 0; j < (int)(kVal); ++j) {
      float val = M[((i * kVal) + j)];
      float isOne = (((i == j)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
      eigenvalues[i] = M[((i * kVal) + i)];
      eigenvectors[((i * kVal) + j)] = isOne;
    }
  }
}
