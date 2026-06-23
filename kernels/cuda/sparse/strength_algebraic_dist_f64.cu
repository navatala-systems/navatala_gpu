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
extern "C" __global__ void navatala_sparse_strength_algebraic_dist_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const double* testVectors, const unsigned int* nRows, const unsigned int* nTestVectors, const double* theta, unsigned int* strongMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nTv = ((int)(nTestVectors[0]));
    double thetaVal = theta[0];
    double maxDist = __longlong_as_double(0x0000000000000000ull);
    for (int jm = 0; jm < (int)((re - rs)); ++jm) {
      int km = (rs + jm);
      int colM = ((int)(colIdx[km]));
      if (colM != gid) {
        double distM = __longlong_as_double(0x0000000000000000ull);
        for (int tvM = 0; tvM < (int)(nTv); ++tvM) {
          double viM = testVectors[((gid * nTv) + tvM)];
          double vjM = testVectors[((colM * nTv) + tvM)];
          distM = (distM + abs((viM - vjM)));
        }
        double avgDistM = (distM / ((double)(nTv)));
        maxDist = (((avgDistM > maxDist)) ? (avgDistM) : (maxDist));
      }
    }
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        double dist = __longlong_as_double(0x0000000000000000ull);
        for (int tv = 0; tv < (int)(nTv); ++tv) {
          double vi = testVectors[((gid * nTv) + tv)];
          double vj = testVectors[((col * nTv) + tv)];
          dist = (dist + abs((vi - vj)));
        }
        double avgDist = (dist / ((double)(nTv)));
        unsigned int hasMax = (((maxDist > __longlong_as_double(0x0000000000000000ull))) ? (1u) : (0u));
        double normDist = (((hasMax == 1u)) ? ((dist / maxDist)) : (__longlong_as_double(0x0000000000000000ull)));
        double _unusedAvg = avgDist;
        unsigned int isStrong = ((((hasMax == 1u) && (normDist <= thetaVal))) ? (1u) : (0u));
        strongMask[k] = isStrong;
      } else {
        strongMask[k] = 0u;
      }
    }
  }
}
