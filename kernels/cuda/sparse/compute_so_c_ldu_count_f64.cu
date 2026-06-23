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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_compute_so_c_ldu_count_f64(const unsigned int* owner, const double* upper, const double* lower, const unsigned int* cellFaceOffsets, const unsigned int* cellFaceIdx, const double* theta, const unsigned int* nCells, unsigned int* nnzPerRow) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    int fStart = ((int)(cellFaceOffsets[cell]));
    int fEnd = ((int)(cellFaceOffsets[(cell + 1)]));
    double maxCoeff = __longlong_as_double(0x0000000000000000ull);
    for (int fi = 0; fi < (int)((fEnd - fStart)); ++fi) {
      int fIdx = (fStart + fi);
      int face = ((int)(cellFaceIdx[fIdx]));
      int own = ((int)(owner[face]));
      double coeff = abs((((cell == own)) ? (upper[face]) : (lower[face])));
      if (coeff > maxCoeff) {
        maxCoeff = coeff;
      }
    }
    double th = theta[0];
    double threshold = (th * maxCoeff);
    int count = 0;
    for (int fi2 = 0; fi2 < (int)((fEnd - fStart)); ++fi2) {
      int fIdx2 = (fStart + fi2);
      int face2 = ((int)(cellFaceIdx[fIdx2]));
      int own2 = ((int)(owner[face2]));
      double coeff2 = abs((((cell == own2)) ? (upper[face2]) : (lower[face2])));
      if (coeff2 > threshold) {
        count = (count + 1);
      }
    }
    nnzPerRow[cell] = ((unsigned int)(count));
  }
}
