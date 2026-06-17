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

__kernel void navatala_cfd_restrict_by_agglomeration_gather_f32(__global const float* fineField, __global const uint* coarseToFineOffsets, __global const uint* coarseToFineList, __global const uint* nCoarseCells, __global float* coarseField) {
  int gid0 = (int)get_global_id(0);
  int c = (int)(get_global_id(0));
  int N = ((int)(nCoarseCells[0]));
  if ((c < N)) {
    int begin = ((int)(coarseToFineOffsets[c]));
    int end = ((int)(coarseToFineOffsets[(c + 1)]));
    float sum = as_float(0x00000000u);
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int fineIdx = ((int)(coarseToFineList[pos]));
      float v = fineField[fineIdx];
      sum = (sum + v);
    }
    coarseField[c] = sum;
  }
}
