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

__kernel void navatala_graph_modularity_matrix_vec_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const float* x, __global const float* degree, __global const float* totalEdgeWeight, __global const uint* n, __global const float* dotDX, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  float dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    float axValAccum = as_float(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float aij = adjacencyValues[idxU32];
        uint j = colInd[idxU32];
        float xj = x[j];
        float product = (aij * xj);
        float currentAxVal = axValAccum;
        float newAxVal = (currentAxVal + product);
        axValAccum = newAxVal;
      }
    }
    float axVal = axValAccum;
    float di = degree[gid];
    float correction = ((di * dotDXVal) / totalWeight);
    float yVal = (axVal - correction);
    y[gid] = yVal;
  }
}
