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

__kernel void navatala_dataframe_tdigest_merge_f32(__global const float* meansA, __global const float* weightsA, __global const uint* countA, __global const float* totalWeightA, __global const float* minA, __global const float* maxA, __global const float* meansB, __global const float* weightsB, __global const uint* countB, __global const float* totalWeightB, __global const float* minB, __global const float* maxB, __global const float* compressionIn, __global float* meansOut, __global float* weightsOut, __global uint* countOut, __global float* totalWeightOut, __global float* minOut, __global float* maxOut, __global float* compressionOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    uint cA = countA[0];
    uint cB = countB[0];
    float twA = totalWeightA[0];
    float twB = totalWeightB[0];
    float minValA = minA[0];
    float maxValA = maxA[0];
    float minValB = minB[0];
    float maxValB = maxB[0];
    float delta = compressionIn[0];
    float newTw = (twA + twB);
    float newMin = (((minValA < minValB)) ? (minValA) : (minValB));
    float newMax = (((maxValA > maxValB)) ? (maxValA) : (maxValB));
    uint newCount = (cA + cB);
    countOut[0] = newCount;
    totalWeightOut[0] = newTw;
    minOut[0] = newMin;
    maxOut[0] = newMax;
    compressionOut[0] = delta;
  }
}
