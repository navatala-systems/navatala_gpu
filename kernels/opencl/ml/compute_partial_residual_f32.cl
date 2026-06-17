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

__kernel void navatala_ml_compute_partial_residual_f32(__global const float* X, __global const float* beta, __global const float* y, __global const uint* j, __global const uint* nSamples, __global const uint* nFeatures, __global float* residuals) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float yVal = y[gid];
    float betaJ = beta[jVal];
    uint rowOffset = (gid * d);
    uint xijIdx = (rowOffset + jVal);
    float xij = X[xijIdx];
    float xijBetaJ = (xij * betaJ);
    float dotAccum = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint xikIdx = (rowOffset + kU32);
      float xik = X[xikIdx];
      float betak = beta[kU32];
      float prod = (xik * betak);
      float currentDot = dotAccum;
      float newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    float dotProduct = dotAccum;
    float diff = (yVal - dotProduct);
    float residVal = (diff + xijBetaJ);
    residuals[gid] = residVal;
  }
}
