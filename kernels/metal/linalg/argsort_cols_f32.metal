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

kernel void navatala_linalg_argsort_cols_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device uint* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float svals[256];
  threadgroup uint sidxs[256];
  uint loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_type<float>(0x00000000u);
  uint loopMyIdx = 0u;
  uint loopPartnerPos = 0u;
  float loopPartnerVal = as_type<float>(0x00000000u);
  uint loopPartnerIdx = 0u;
  uint colIdx = ((uint)(int(__tgid.x)));
  uint rowIdx = ((uint)(int(__tid.x)));
  uint mVal = m[0u];
  uint nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = as_type<float>(0x7f7fffffu);
      sidxs[rowIdx] = rowIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % 2u);
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + 1u) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        float myValRead = loopMyVal;
        loopMyIdx = sidxs[rowIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + 1u);
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}
