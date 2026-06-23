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

__kernel void navatala_sparse_aggregate_parallel_greedy_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const uint* strongMask, __global const uint* nRows, __global const uint* seed, __global int* aggregateId, __global uint* nAggregates) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int bestCol = -1;
    float bestVal = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == (uint)(1u)) {
        float a = values[k];
        if (fabs(a) > bestVal) {
          bestVal = fabs(a);
          bestCol = ((int)(colIdx[k]));
        }
      }
    }
    if (bestCol >= 0) {
      int aggVal = (((gid < bestCol)) ? (gid) : (bestCol));
      aggregateId[gid] = aggVal;
      aggregateId[bestCol] = aggVal;
    } else {
      aggregateId[gid] = gid;
    }
  }
}
