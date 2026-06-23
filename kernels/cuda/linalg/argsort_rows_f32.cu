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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_argsort_rows_f32(const float* A, const unsigned int* m, const unsigned int* n, unsigned int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float svals[256];
  __shared__ unsigned int sidxs[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = __uint_as_float(0x00000000u);
  unsigned int loopMyIdx = 0u;
  unsigned int loopPartnerPos = 0u;
  float loopPartnerVal = __uint_as_float(0x00000000u);
  unsigned int loopPartnerIdx = 0u;
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int colIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
      sidxs[colIdx] = colIdx;
    } else {
      svals[colIdx] = __uint_as_float(0x7f7fffffu);
      sidxs[colIdx] = colIdx;
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (colIdx % 2u);
      unsigned int myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + 1u) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        float myValRead = loopMyVal;
        loopMyIdx = sidxs[colIdx];
        unsigned int myIdxRead = loopMyIdx;
        loopPartnerPos = (colIdx + 1u);
        unsigned int partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        unsigned int partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          sidxs[colIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      unsigned int sortedIdx = sidxs[colIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}
