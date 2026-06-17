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

__kernel void navatala_sparse_strength_algebraic_dist_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* testVectors, __global const uint* nRows, __global const uint* nTestVectors, __global const float* theta, __global uint* strongMask) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nTv = ((int)(nTestVectors[0]));
    float thetaVal = theta[0];
    float maxDist = as_float(0x00000000u);
    for (int jm = 0; jm < (int)((re - rs)); ++jm) {
      int km = (rs + jm);
      int colM = ((int)(colIdx[km]));
      if ((colM != gid)) {
        float distM = as_float(0x00000000u);
        for (int tvM = 0; tvM < (int)(nTv); ++tvM) {
          float viM = testVectors[((gid * nTv) + tvM)];
          float vjM = testVectors[((colM * nTv) + tvM)];
          distM = (distM + fabs((viM - vjM)));
        }
        float avgDistM = (distM / ((float)(nTv)));
        maxDist = (((avgDistM > maxDist)) ? (avgDistM) : (maxDist));
      }
    }
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if ((col != gid)) {
        float dist = as_float(0x00000000u);
        for (int tv = 0; tv < (int)(nTv); ++tv) {
          float vi = testVectors[((gid * nTv) + tv)];
          float vj = testVectors[((col * nTv) + tv)];
          dist = (dist + fabs((vi - vj)));
        }
        float avgDist = (dist / ((float)(nTv)));
        uint hasMax = (((maxDist > as_float(0x00000000u))) ? ((uint)(1u)) : ((uint)(0u)));
        float normDist = (((hasMax == (uint)(1u))) ? ((dist / maxDist)) : (as_float(0x00000000u)));
        float _unusedAvg = avgDist;
        uint isStrong = ((((hasMax == (uint)(1u)) && (normDist <= thetaVal))) ? ((uint)(1u)) : ((uint)(0u)));
        strongMask[k] = isStrong;
      } else {
        strongMask[k] = (uint)(0u);
      }
    }
  }
}
