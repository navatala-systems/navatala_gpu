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

kernel void navatala_linalg_transpose_in_place_f32(device const uint* n [[buffer(0)]], device float* A [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint dim = n[0];
  uint numSwaps = ((dim * (dim - 1u)) / 2u);
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    float gidF = ((float)(gid));
    float dimF = ((float)(dim));
    float twoNm1 = ((as_type<float>(0x40000000u) * dimF) - as_type<float>(0x3f800000u));
    float disc = ((twoNm1 * twoNm1) - (as_type<float>(0x41000000u) * gidF));
    float sqrtDisc = sqrt(disc);
    float iF = floor(((twoNm1 - sqrtDisc) / as_type<float>(0x40000000u)));
    uint i = ((uint)(iF));
    uint rowStart = ((i * (((2u * dim) - i) - 1u)) / 2u);
    uint offset = (gid - rowStart);
    uint j = ((i + 1u) + offset);
    uint idx_ij = ((i * dim) + j);
    uint idx_ji = ((j * dim) + i);
    float val_ij = A[idx_ij];
    float val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}
