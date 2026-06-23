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
extern "C" __global__ void navatala_linalg_sort_cols_f32(const float* A, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float svals[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = __uint_as_float(0x00000000u);
  unsigned int loopPartnerIdx = 0u;
  float loopPartnerVal = __uint_as_float(0x00000000u);
  unsigned int colIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
    } else {
      svals[rowIdx] = __uint_as_float(0x7f7fffffu);
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (rowIdx % 2u);
      unsigned int myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + 1u) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        float myValRead = loopMyVal;
        loopPartnerIdx = (rowIdx + 1u);
        unsigned int partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      float sortedVal = svals[rowIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}
