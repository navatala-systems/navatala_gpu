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

__kernel void navatala_sparse_chebyshev_smooth_f32(__global const uint* rowPtr, __global const uint* colInd, __global const float* values, __global const float* x, __global const float* xPrev, __global const float* diagInv, __global const float* b, __global const uint* n, __global const float* alpha, __global const float* beta, __global float* xNext) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint i = gid;
  uint nVal = n[(uint)(0u)];
  if ((i < nVal)) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + (uint)(1u));
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_float(0x00000000u);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if ((jU32 >= rowStart)) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        float xj = x[col];
        float term = (aVal * xj);
        float currentSum = sumAccum;
        sumAccum = (currentSum + term);
      }
    }
    float bi = b[i];
    float xi = x[i];
    float xPrevI = xPrev[i];
    float dInv = diagInv[i];
    float alphaVal = alpha[(uint)(0u)];
    float betaVal = beta[(uint)(0u)];
    float residual = (bi - sumAccum);
    float scaledResidual = (dInv * residual);
    float history = (xi - xPrevI);
    float update = ((alphaVal * scaledResidual) + (betaVal * history));
    xNext[i] = (xi + update);
  }
}
