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

__kernel void navatala_ml_add_regularization_f32(__global float* G, __global const float* lambda, __global const uint* p) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint pVal = p[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    uint diagIdx = ((gid * pVal) + gid);
    float currentVal = G[diagIdx];
    float newVal = (currentVal + lambdaVal);
    G[diagIdx] = newVal;
  }
}
