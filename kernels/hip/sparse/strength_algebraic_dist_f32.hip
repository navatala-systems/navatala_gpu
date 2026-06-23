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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_strength_algebraic_dist_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* testVectors, const unsigned int* nRows, const unsigned int* nTestVectors, const float* theta, unsigned int* strongMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nTv = ((int)(nTestVectors[0]));
    float thetaVal = theta[0];
    float maxDist = __uint_as_float(0x00000000u);
    for (int jm = 0; jm < (int)((re - rs)); ++jm) {
      int km = (rs + jm);
      int colM = ((int)(colIdx[km]));
      if (colM != gid) {
        float distM = __uint_as_float(0x00000000u);
        for (int tvM = 0; tvM < (int)(nTv); ++tvM) {
          float viM = testVectors[((gid * nTv) + tvM)];
          float vjM = testVectors[((colM * nTv) + tvM)];
          distM = (distM + abs((viM - vjM)));
        }
        float avgDistM = (distM / ((float)(nTv)));
        maxDist = (((avgDistM > maxDist)) ? (avgDistM) : (maxDist));
      }
    }
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        float dist = __uint_as_float(0x00000000u);
        for (int tv = 0; tv < (int)(nTv); ++tv) {
          float vi = testVectors[((gid * nTv) + tv)];
          float vj = testVectors[((col * nTv) + tv)];
          dist = (dist + abs((vi - vj)));
        }
        float avgDist = (dist / ((float)(nTv)));
        unsigned int hasMax = (((maxDist > __uint_as_float(0x00000000u))) ? (1u) : (0u));
        float normDist = (((hasMax == 1u)) ? ((dist / maxDist)) : (__uint_as_float(0x00000000u)));
        float _unusedAvg = avgDist;
        unsigned int isStrong = ((((hasMax == 1u) && (normDist <= thetaVal))) ? (1u) : (0u));
        strongMask[k] = isStrong;
      } else {
        strongMask[k] = 0u;
      }
    }
  }
}
