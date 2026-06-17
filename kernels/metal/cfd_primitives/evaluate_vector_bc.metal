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

kernel void navatala_cfd_primitives_evaluate_vector_bc(device const float* inX [[buffer(0)]], device const float* inY [[buffer(1)]], device const float* inZ [[buffer(2)]], device const int* faceCells [[buffer(3)]], device const uint* bcTypeMask [[buffer(4)]], device const float* fvX [[buffer(5)]], device const float* fvY [[buffer(6)]], device const float* fvZ [[buffer(7)]], device const uint* counts [[buffer(8)]], device float* outX [[buffer(9)]], device float* outY [[buffer(10)]], device float* outZ [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(counts[0])))) {
    return;
  } else {
    uint bcType = bcTypeMask[int(__gid.x)];
    int cell = faceCells[int(__gid.x)];
    if ((((int)(bcType)) == 1)) {
      outX[int(__gid.x)] = fvX[int(__gid.x)];
      outY[int(__gid.x)] = fvY[int(__gid.x)];
      outZ[int(__gid.x)] = fvZ[int(__gid.x)];
    } else {
      outX[int(__gid.x)] = inX[cell];
      outY[int(__gid.x)] = inY[cell];
      outZ[int(__gid.x)] = inZ[cell];
    }
  }
}
