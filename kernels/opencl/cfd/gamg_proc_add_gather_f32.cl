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

__kernel void navatala_cfd_gamg_proc_add_gather_f32(__global const uint* entryOffsets, __global const uint* entryIndices, __global const float* coeffs, __global const float* recvVals, __global const uint* nCells, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int c = (int)(get_global_id(0));
  int N = ((int)(nCells[0]));
  if ((c < N)) {
    int begin = ((int)(entryOffsets[c]));
    int end = ((int)(entryOffsets[(c + 1)]));
    float acc = y[c];
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int entry = ((int)(entryIndices[pos]));
      float co = coeffs[entry];
      float rv = recvVals[entry];
      acc = (acc + (co * rv));
    }
    y[c] = acc;
  }
}
