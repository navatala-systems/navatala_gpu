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

__kernel void navatala_ml_make_regression_f32(__global const float* X, __global const float* w, __global const float* bias, __global const float* noise, __global const uint* n, __global const uint* d, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  uint numD = d[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float biasVal = bias[0];
    uint sampleIdx = gid;
    uint baseIdx = (sampleIdx * numD);
    float x0 = X[baseIdx];
    float w0 = w[0];
    float dot = (x0 * w0);
    float noiseVal = noise[gid];
    float withBias = (dot + biasVal);
    float result = (withBias + noiseVal);
    y[gid] = result;
  }
}
