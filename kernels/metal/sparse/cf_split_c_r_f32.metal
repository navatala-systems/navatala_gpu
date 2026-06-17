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

kernel void navatala_sparse_cf_split_c_r_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const float* crThreshold [[buffer(5)]], device int* cfMarking [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int mark = cfMarking[row];
    if ((mark == 0)) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      float diag = as_type<float>(0x00000000u);
      float offDiagSum = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        float a = values[k];
        if ((col == row)) {
          diag = a;
        } else {
          offDiagSum = (offDiagSum + a);
        }
      }
      float crVal = abs((offDiagSum / diag));
      float crThresh = crThreshold[0];
      if ((crVal > crThresh)) {
        cfMarking[row] = 1;
      } else {
        bool hasC = false;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          int k2 = (rs + j2);
          uint isStr = strongMask[k2];
          if ((isStr == 1u)) {
            int nbr = ((int)(colIdx[k2]));
            int nbrM = cfMarking[nbr];
            if ((nbrM == 1)) {
              hasC = true;
            }
          }
        }
        if (hasC) {
          cfMarking[row] = -1;
        }
      }
    }
  }
}
