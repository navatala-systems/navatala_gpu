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

kernel void navatala_sparse_knn_sparse_top_k_f32(device const float* distances [[buffer(0)]], device const uint* nIdxRows [[buffer(1)]], device const uint* k [[buffer(2)]], device float* outDistances [[buffer(3)]], device uint* outIndices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    uint nRows = nIdxRows[0];
    uint kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      outDistances[initIdxU32] = as_type<float>(0x7f7fc99eu);
      outIndices[initIdxU32] = 4294967295u;
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      uint iU32 = ((uint)(i));
      float dist = distances[iU32];
      uint lastIdx = (kVal - 1u);
      float kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          float checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          uint newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            uint srcPos = (shiftPos - 1u);
            float srcDist = outDistances[srcPos];
            uint srcIdx = outIndices[srcPos];
            outDistances[shiftPos] = srcDist;
            outIndices[shiftPos] = srcIdx;
          }
        }
        outDistances[finalInsertPos] = dist;
        outIndices[finalInsertPos] = iU32;
      }
    }
  }
}
