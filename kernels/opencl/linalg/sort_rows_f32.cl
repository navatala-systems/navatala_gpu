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

__kernel void navatala_linalg_sort_rows_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float svals[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_float(0x00000000u);
  uint loopPartnerIdx = (uint)(0u);
  float loopPartnerVal = as_float(0x00000000u);
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint colIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = as_float(0x7f7fffffu);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + (uint)(1u)) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        float myValRead = loopMyVal;
        loopPartnerIdx = (colIdx + (uint)(1u));
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      float sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}
