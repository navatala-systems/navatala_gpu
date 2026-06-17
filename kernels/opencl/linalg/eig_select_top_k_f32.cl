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

__kernel void navatala_linalg_eig_select_top_k_f32(__global const float* eigenvalues, __global const float* eigenvectors, __global const uint* k, __global const uint* n, __global float* topKValues, __global float* topKVectors) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint kVal = k[0];
  uint nVal = n[0];
  uint totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint col = (gid / nVal);
    uint row = (gid % nVal);
    bool isFirstRow = (row == (uint)(0u));
    if (isFirstRow) {
      float eigenval = eigenvalues[col];
      topKValues[col] = eigenval;
    }
    uint srcIdx = ((col * nVal) + row);
    float vecElem = eigenvectors[srcIdx];
    topKVectors[srcIdx] = vecElem;
  }
}
