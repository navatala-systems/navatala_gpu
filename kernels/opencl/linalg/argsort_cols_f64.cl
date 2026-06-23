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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_argsort_cols_f64(__global const double* A, __global const uint* m, __global const uint* n, __global uint* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local double svals[256];
  __local uint sidxs[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = as_double(0x0000000000000000ul);
  uint loopMyIdx = (uint)(0u);
  uint loopPartnerPos = (uint)(0u);
  double loopPartnerVal = as_double(0x0000000000000000ul);
  uint loopPartnerIdx = (uint)(0u);
  uint colIdx = ((uint)((int)(get_group_id(0))));
  uint rowIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = as_double(0x7feffffffffffffful);
      sidxs[rowIdx] = rowIdx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + (uint)(1u)) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        double myValRead = loopMyVal;
        loopMyIdx = sidxs[rowIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + (uint)(1u));
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        double partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}
