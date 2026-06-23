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

kernel void navatala_sparse_interpolate_distance1_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const uint* ProwPtr [[buffer(5)]], device const uint* nRows [[buffer(6)]], device uint* PcolIdx [[buffer(7)]], device float* Pvalues [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int mark = cfMarking[row];
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_type<float>(0x3f800000u);
    } else {
      if (mark == -1) {
        float diag = as_type<float>(0x00000000u);
        for (int jd = 0; jd < (int)((re - rs)); ++jd) {
          int kd = (rs + jd);
          int colD = ((int)(colIdx[kd]));
          if (colD == row) {
            diag = values[kd];
          }
        }
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int markW = cfMarking[colW];
            if (markW == 1) {
              float aij = values[kw];
              float w = (-(aij / diag));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
      }
    }
  }
}
