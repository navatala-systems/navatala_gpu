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

__kernel void navatala_ml_pairwise_cosine_f32(__global const float* X, __global const float* Y, __global const uint* n, __global const uint* m, __global const uint* d, __global float* D) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float dotProduct = as_float(0x00000000u);
    float normXSq = as_float(0x00000000u);
    float normYSq = as_float(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float prod = (xVal * yVal);
      float currentDot = dotProduct;
      float newDot = (currentDot + prod);
      dotProduct = newDot;
      float xSq = (xVal * xVal);
      float ySq = (yVal * yVal);
      float currentNormX = normXSq;
      float newNormX = (currentNormX + xSq);
      normXSq = newNormX;
      float currentNormY = normYSq;
      float newNormY = (currentNormY + ySq);
      normYSq = newNormY;
    }
    float finalDot = dotProduct;
    float finalNormXSq = normXSq;
    float finalNormYSq = normYSq;
    float normX = sqrt(finalNormXSq);
    float normY = sqrt(finalNormYSq);
    float normProduct = (normX * normY);
    bool normIsZero = (normProduct == as_float(0x00000000u));
    if (normIsZero) {
      D[gid] = as_float(0x00000000u);
    } else {
      float cosineSim = (finalDot / normProduct);
      float cosineDist = (as_float(0x3f800000u) - cosineSim);
      D[gid] = cosineDist;
    }
  }
}
