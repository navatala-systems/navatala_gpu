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
__kernel void navatala_graph_modularity_matrix_vec_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const double* x, __global const double* degree, __global const double* totalEdgeWeight, __global const uint* n, __global const double* dotDX, __global double* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  double dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double axValAccum = as_double(0x0000000000000000ul);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double aij = adjacencyValues[idxU32];
        uint j = colInd[idxU32];
        double xj = x[j];
        double product = (aij * xj);
        double currentAxVal = axValAccum;
        double newAxVal = (currentAxVal + product);
        axValAccum = newAxVal;
      }
    }
    double axVal = axValAccum;
    double di = degree[gid];
    double correction = ((di * dotDXVal) / totalWeight);
    double yVal = (axVal - correction);
    y[gid] = yVal;
  }
}
