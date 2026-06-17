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

__kernel void navatala_dataframe_tdigest_add_f32(__global const float* value, __global const float* weight, __global float* centroidMeans, __global float* centroidWeights, __global uint* centroidCount, __global float* totalWeight, __global float* minVal, __global float* maxVal) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    float x = value[0];
    float w = weight[0];
    uint count = centroidCount[0];
    float tw = totalWeight[0];
    float minV = minVal[0];
    float maxV = maxVal[0];
    float newMin = (((x < minV)) ? (x) : (minV));
    float newMax = (((x > maxV)) ? (x) : (maxV));
    float newTw = (tw + w);
    uint newCount = (count + (uint)(1u));
    uint idx = count;
    centroidMeans[((int)(idx))] = x;
    centroidWeights[((int)(idx))] = w;
    centroidCount[0] = newCount;
    totalWeight[0] = newTw;
    minVal[0] = newMin;
    maxVal[0] = newMax;
  }
}
