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
extern "C" __global__ void navatala_linalg_sort_rows_f64(const double* A, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double svals[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopPartnerIdx = 0u;
  double loopPartnerVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int colIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = __longlong_as_double(0x7fefffffffffffffull);
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
        double myValRead = loopMyVal;
        loopPartnerIdx = (colIdx + 1u);
        unsigned int partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        double partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      double sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}
