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

kernel void navatala_cfd_primitives_div_face_flux_vector_csr(device const int* offsets [[buffer(0)]], device const int* faceIdx [[buffer(1)]], device const float* sign [[buffer(2)]], device const float* faceFluxX [[buffer(3)]], device const float* faceFluxY [[buffer(4)]], device const float* faceFluxZ [[buffer(5)]], device const float* vol [[buffer(6)]], device const int* params [[buffer(7)]], device float* outX [[buffer(8)]], device float* outY [[buffer(9)]], device float* outZ [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float accX = as_type<float>(0x00000000u);
    float accY = as_type<float>(0x00000000u);
    float accZ = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      accX = (accX + (s * faceFluxX[f]));
      accY = (accY + (s * faceFluxY[f]));
      accZ = (accZ + (s * faceFluxZ[f]));
    }
    float volCell = vol[((int)(int(__gid.x)))];
    float volSafe = (((abs(volCell) > as_type<float>(0x1e3ce508u))) ? (volCell) : (as_type<float>(0x1e3ce508u)));
    float invV = (as_type<float>(0x3f800000u) / volSafe);
    outX[((int)(int(__gid.x)))] = (accX * invV);
    outY[((int)(int(__gid.x)))] = (accY * invV);
    outZ[((int)(int(__gid.x)))] = (accZ * invV);
  }
}
