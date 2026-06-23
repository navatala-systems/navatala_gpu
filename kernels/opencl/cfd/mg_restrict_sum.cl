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

__kernel void navatala_cfd_mg_restrict_sum(__global const float* rFine, __global const int* prolongRowOffsets, __global const int* prolongCols, __global const float* prolongVals, __global float* rCoarse, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    int beg = prolongRowOffsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    int end = prolongRowOffsets[c1];
    float v = rFine[(int)(get_global_id(0))];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      atomic_add(&rCoarse[col], (prolongVals[k] * v));
    }
  }
}
