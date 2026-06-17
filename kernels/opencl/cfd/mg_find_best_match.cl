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

__kernel void navatala_cfd_mg_find_best_match(__global const uint* csrOffsets, __global const uint* csrFaceIdx, __global const int* csrSign, __global const float* strength, __global const int* owner, __global const int* nei, __global int* bestNeighbor, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    float bestS = as_float(0xbf800000u);
    int bestN = -1;
    uint beg = csrOffsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = csrOffsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = csrFaceIdx[k];
      float s = strength[((int)(f))];
      if ((s > bestS)) {
        bestS = s;
        int sg = csrSign[k];
        if ((sg > 0)) {
          bestN = nei[((int)(f))];
        } else {
          bestN = owner[((int)(f))];
        }
      }
    }
    bestNeighbor[(int)(get_global_id(0))] = bestN;
  }
}
