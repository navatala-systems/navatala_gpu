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

__kernel void navatala_graph_laplacian_from_adjacency_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* n, __global float* laplacianValues, __global float* degree) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[rowIdx];
    uint one32 = ((uint)(1));
    uint rowIdxPlusOne = (rowIdx + one32);
    uint rowEnd = rowPtr[rowIdxPlusOne];
    float degreeSumAccum = as_float(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      uint jU32 = ((uint)(j));
      uint absIdx = (rowStart + jU32);
      float adjVal = adjacencyValues[absIdx];
      float negAdjVal = (as_float(0x00000000u) - adjVal);
      laplacianValues[absIdx] = negAdjVal;
      float currentDegree = degreeSumAccum;
      float newDegree = (currentDegree + adjVal);
      degreeSumAccum = newDegree;
    }
    float finalDegree = degreeSumAccum;
    degree[rowIdx] = finalDegree;
  }
}
