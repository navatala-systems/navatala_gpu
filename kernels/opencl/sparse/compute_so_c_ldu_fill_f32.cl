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

__kernel void navatala_sparse_compute_so_c_ldu_fill_f32(__global const uint* owner, __global const uint* neighbour, __global const float* upper, __global const float* lower, __global const uint* cellFaceOffsets, __global const uint* cellFaceIdx, __global const uint* strongRowPtr, __global const float* theta, __global const uint* nCells, __global uint* strongColIdx) {
  int gid0 = (int)get_global_id(0);
  int cell = (int)(get_global_id(0));
  int N = ((int)(nCells[0]));
  if (cell < N) {
    int fStart = ((int)(cellFaceOffsets[cell]));
    int fEnd = ((int)(cellFaceOffsets[(cell + 1)]));
    float maxCoeff = as_float(0x00000000u);
    for (int fi = 0; fi < (int)((fEnd - fStart)); ++fi) {
      int fIdx = (fStart + fi);
      int face = ((int)(cellFaceIdx[fIdx]));
      int own = ((int)(owner[face]));
      float coeff = fabs((((cell == own)) ? (upper[face]) : (lower[face])));
      if (coeff > maxCoeff) {
        maxCoeff = coeff;
      }
    }
    int rowStart = ((int)(strongRowPtr[cell]));
    float th = theta[0];
    float threshold = (th * maxCoeff);
    int off = 0;
    for (int fi2 = 0; fi2 < (int)((fEnd - fStart)); ++fi2) {
      int fIdx2 = (fStart + fi2);
      int face2 = ((int)(cellFaceIdx[fIdx2]));
      int own2 = ((int)(owner[face2]));
      float coeff2 = fabs((((cell == own2)) ? (upper[face2]) : (lower[face2])));
      if (coeff2 > threshold) {
        int nbr2 = ((int)((((cell == own2)) ? (neighbour[face2]) : (owner[face2]))));
        strongColIdx[(rowStart + off)] = ((uint)(nbr2));
        off = (off + 1);
      }
    }
  }
}
