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

__kernel void navatala_sparse_sparse_gauss_seidel_iter_f32(__global const uint* rowPtr, __global const uint* colInd, __global const float* values, __global const float* x, __global const float* b, __global const uint* n, __global float* x_new) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint i = gid;
  uint nVal = n[(uint)(0u)];
  if ((i < nVal)) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + (uint)(1u));
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_float(0x00000000u);
    float diagAccum = as_float(0x3f800000u);
    float bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if ((jU32 >= rowStart)) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        bool isDiag = (col == i);
        bool isLower = (col < i);
        float currentDiag = diagAccum;
        float newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        float xjNew = x_new[col];
        float xjOld = x[col];
        float xj = ((isLower) ? (xjNew) : (xjOld));
        float contrib = (aVal * xj);
        float offDiagContrib = ((isDiag) ? (as_float(0x00000000u)) : (contrib));
        float currentSum = sumAccum;
        float newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    float finalSum = sumAccum;
    float finalDiag = diagAccum;
    float numerator = (bi - finalSum);
    float result = (numerator / finalDiag);
    x_new[i] = result;
  }
}
