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

__kernel void navatala_ml_compute_correlations_f32(__global const float* X, __global const float* residual, __global const uint* nRows, __global const uint* nCols, __global float* correlations) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nRows[0];
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint colOffset = (gid * n);
    float accum = as_float(0x00000000u);
    for (int i = 0; i < (int)(n); ++i) {
      uint iU32 = ((uint)(i));
      uint xIdx = (iU32 + colOffset);
      float xVal = X[xIdx];
      float rVal = residual[iU32];
      float prod = (xVal * rVal);
      float currentAccum = accum;
      float newAccum = (currentAccum + prod);
      accum = newAccum;
    }
    float corr = accum;
    correlations[gid] = corr;
  }
}
