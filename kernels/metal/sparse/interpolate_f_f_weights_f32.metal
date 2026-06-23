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

kernel void navatala_sparse_interpolate_f_f_weights_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const float* diag [[buffer(5)]], device const uint* ProwPtr [[buffer(6)]], device const uint* PcolIdx [[buffer(7)]], device const uint* nRows [[buffer(8)]], device float* Pvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == -1) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      float diagVal = diag[row];
      int pStart = ((int)(ProwPtr[row]));
      int pos = 0;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == 1u) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            float aij = values[k];
            float numer = aij;
            for (int ff = 0; ff < (int)((re - rs)); ++ff) {
              int kf = (rs + ff);
              uint isStrF = strongMask[kf];
              if (isStrF == 1u) {
                int colF = ((int)(colIdx[kf]));
                int mF = cfMarking[colF];
                if (mF == -1) {
                  float aik = values[kf];
                  float diagK = diag[colF];
                  int rsK = ((int)(rowPtr[colF]));
                  int reK = ((int)(rowPtr[(colF + 1)]));
                  float akj = as_type<float>(0x00000000u);
                  for (int kn = 0; kn < (int)((reK - rsK)); ++kn) {
                    int knIdx = (rsK + kn);
                    int knCol = ((int)(colIdx[knIdx]));
                    if (knCol == col) {
                      akj = values[knIdx];
                    }
                  }
                  numer = (numer + (aik * (akj / diagK)));
                }
              }
            }
            float w = (-(numer / diagVal));
            Pvalues[(pStart + pos)] = w;
            pos = (pos + 1);
          }
        }
      }
    }
  }
}
