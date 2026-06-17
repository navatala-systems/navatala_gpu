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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_tdigest_merge_f32(device const float* meansA [[buffer(0)]], device const float* weightsA [[buffer(1)]], device const uint* countA [[buffer(2)]], device const float* totalWeightA [[buffer(3)]], device const float* minA [[buffer(4)]], device const float* maxA [[buffer(5)]], device const float* meansB [[buffer(6)]], device const float* weightsB [[buffer(7)]], device const uint* countB [[buffer(8)]], device const float* totalWeightB [[buffer(9)]], device const float* minB [[buffer(10)]], device const float* maxB [[buffer(11)]], device const float* compressionIn [[buffer(12)]], device float* meansOut [[buffer(13)]], device float* weightsOut [[buffer(14)]], device uint* countOut [[buffer(15)]], device float* totalWeightOut [[buffer(16)]], device float* minOut [[buffer(17)]], device float* maxOut [[buffer(18)]], device float* compressionOut [[buffer(19)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if ((gid == 0u)) {
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
