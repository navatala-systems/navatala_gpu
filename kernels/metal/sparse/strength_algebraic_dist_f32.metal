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

kernel void navatala_sparse_strength_algebraic_dist_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* testVectors [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* nTestVectors [[buffer(5)]], device const float* theta [[buffer(6)]], device uint* strongMask [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nTv = ((int)(nTestVectors[0]));
    float thetaVal = theta[0];
    float maxDist = as_type<float>(0x00000000u);
    for (int jm = 0; jm < (int)((re - rs)); ++jm) {
      int km = (rs + jm);
      int colM = ((int)(colIdx[km]));
      if (colM != gid) {
        float distM = as_type<float>(0x00000000u);
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
        float dist = as_type<float>(0x00000000u);
        for (int tv = 0; tv < (int)(nTv); ++tv) {
          float vi = testVectors[((gid * nTv) + tv)];
          float vj = testVectors[((col * nTv) + tv)];
          dist = (dist + abs((vi - vj)));
        }
        float avgDist = (dist / ((float)(nTv)));
        uint hasMax = (((maxDist > as_type<float>(0x00000000u))) ? (1u) : (0u));
        float normDist = (((hasMax == 1u)) ? ((dist / maxDist)) : (as_type<float>(0x00000000u)));
        float _unusedAvg = avgDist;
        uint isStrong = ((((hasMax == 1u) && (normDist <= thetaVal))) ? (1u) : (0u));
        strongMask[k] = isStrong;
      } else {
        strongMask[k] = 0u;
      }
    }
  }
}
