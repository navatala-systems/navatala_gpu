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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_laplacian_from_adjacency_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* n, __global double* laplacianValues, __global double* degree) {
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
    double degreeSumAccum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      uint jU32 = ((uint)(j));
      uint absIdx = (rowStart + jU32);
      double adjVal = adjacencyValues[absIdx];
      double negAdjVal = (as_double(0x0000000000000000ul) - adjVal);
      laplacianValues[absIdx] = negAdjVal;
      double currentDegree = degreeSumAccum;
      double newDegree = (currentDegree + adjVal);
      degreeSumAccum = newDegree;
    }
    double finalDegree = degreeSumAccum;
    degree[rowIdx] = finalDegree;
  }
}
