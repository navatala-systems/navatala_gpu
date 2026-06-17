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

kernel void navatala_dataframe_arg_top_k_per_row_f32(device const float* values [[buffer(0)]], device const uint* q [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device uint* outIndices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint kVal = k[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < qVal);
  if (inBounds) {
    uint inBase = (rowIdx * nVal);
    uint outBase = (rowIdx * kVal);
    float topKVal0 = as_type<float>(0x7f7fc99eu);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      outIndices[outIdx] = 4294967295u;
    }
    for (int col = 0; col < (int)(nVal); ++col) {
      uint colU32 = ((uint)(col));
      uint inIdx = (inBase + colU32);
      float val = values[inIdx];
      uint lastIdx = (kVal - 1u);
      uint lastOutIdx = (outBase + lastIdx);
      uint kthIdx = outIndices[lastOutIdx];
      bool kthIdxValid = (kthIdx != 4294967295u);
      uint kthInIdx = (inBase + kthIdx);
      float kthValIfValid = values[kthInIdx];
      float kthVal = ((kthIdxValid) ? (kthValIfValid) : (as_type<float>(0x7f7fc99eu)));
      bool shouldInsert = (val < kthVal);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          uint checkOutIdx = (outBase + checkPos);
          uint checkIdx = outIndices[checkOutIdx];
          bool checkIdxValid = (checkIdx != 4294967295u);
          uint checkInIdx = (inBase + checkIdx);
          float checkValIfValid = values[checkInIdx];
          float checkVal = ((checkIdxValid) ? (checkValIfValid) : (as_type<float>(0x7f7fc99eu)));
          bool isSmaller = (val < checkVal);
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
            uint srcOutIdx = (outBase + srcPos);
            uint dstOutIdx = (outBase + shiftPos);
            uint srcIdx = outIndices[srcOutIdx];
            outIndices[dstOutIdx] = srcIdx;
          }
        }
        uint insertOutIdx = (outBase + finalInsertPos);
        outIndices[insertOutIdx] = colU32;
      }
    }
  }
}
