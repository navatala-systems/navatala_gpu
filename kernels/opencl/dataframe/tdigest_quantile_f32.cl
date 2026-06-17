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

__kernel void navatala_dataframe_tdigest_quantile_f32(__global const float* p, __global const float* centroidMeans, __global const float* centroidWeights, __global const uint* centroidCount, __global const float* totalWeight, __global const float* minVal, __global const float* maxVal, __global float* quantile) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    float pVal = p[0];
    uint count = centroidCount[0];
    float minV = minVal[0];
    float maxV = maxVal[0];
    float range = (maxV - minV);
    float result = (minV + (pVal * range));
    bool isEmpty = (count == (uint)(0u));
    float finalResult = ((isEmpty) ? (as_float(0x00000000u)) : (result));
    quantile[0] = finalResult;
  }
}
