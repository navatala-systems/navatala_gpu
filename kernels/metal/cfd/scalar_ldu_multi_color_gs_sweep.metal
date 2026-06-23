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

kernel void navatala_cfd_scalar_ldu_multi_color_gs_sweep(device const float* diag [[buffer(0)]], device const float* upper [[buffer(1)]], device const float* lower [[buffer(2)]], device const float* rhs [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* neighbour [[buffer(5)]], device const uint* offsets [[buffer(6)]], device const uint* faceIdx [[buffer(7)]], device const int* sign [[buffer(8)]], device const uint* colorOrder [[buffer(9)]], device const uint* colorRange [[buffer(10)]], device const float* params [[buffer(11)]], device const uint* counts [[buffer(12)]], device float* x [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int colorSize = (((int)(colorRange[1])) - ((int)(colorRange[0])));
  if (int(__gid.x) >= colorSize) {
    return;
  } else {
    int cellIdx = (((int)(colorRange[0])) + int(__gid.x));
    uint celli = colorOrder[cellIdx];
    int ci = ((int)(celli));
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[ci];
    float absD = abs(d);
    if (absD < smallDiag) {
      if (d >= as_type<float>(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float s = (d * x[ci]);
    int beg = ((int)(offsets[ci]));
    int c1 = (ci + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint f = faceIdx[k];
      int sg = sign[k];
      if (((int)(f)) < ((int)(counts[1]))) {
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    x[ci] = (x[ci] + (omega * ((rhs[ci] - s) / d)));
  }
}
