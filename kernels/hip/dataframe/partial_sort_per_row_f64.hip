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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_partial_sort_per_row_f64(const double* values, const unsigned int* q, const unsigned int* n, const unsigned int* k, double* outValues, unsigned int* outIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int qVal = q[0];
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < qVal);
  if (inBounds) {
    unsigned int inBase = (rowIdx * nVal);
    unsigned int outBase = (rowIdx * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      unsigned int initIdxU32 = ((unsigned int)(initIdx));
      unsigned int outIdx = (outBase + initIdxU32);
      outValues[outIdx] = __longlong_as_double(0x7fee42d130773b76ull);
      outIndices[outIdx] = 4294967295u;
    }
    for (int col = 0; col < (int)(nVal); ++col) {
      unsigned int colU32 = ((unsigned int)(col));
      unsigned int inIdx = (inBase + colU32);
      double val = values[inIdx];
      unsigned int lastIdx = (kVal - 1u);
      unsigned int lastOutIdx = (outBase + lastIdx);
      double kthVal = outValues[lastOutIdx];
      bool shouldInsert = (val < kthVal);
      if (shouldInsert) {
        unsigned int insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          unsigned int scanIdxU32 = ((unsigned int)(scanIdx));
          unsigned int currentInsertPos = insertPosAccum;
          unsigned int checkPos = (lastIdx - scanIdxU32);
          unsigned int checkOutIdx = (outBase + checkPos);
          double checkVal = outValues[checkOutIdx];
          bool isSmaller = (val < checkVal);
          unsigned int newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        unsigned int finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          unsigned int shiftIdxU32 = ((unsigned int)(shiftIdx));
          unsigned int shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            unsigned int srcPos = (shiftPos - 1u);
            unsigned int srcOutIdx = (outBase + srcPos);
            unsigned int dstOutIdx = (outBase + shiftPos);
            double srcVal = outValues[srcOutIdx];
            unsigned int srcIdx = outIndices[srcOutIdx];
            outValues[dstOutIdx] = srcVal;
            outIndices[dstOutIdx] = srcIdx;
          }
        }
        unsigned int insertOutIdx = (outBase + finalInsertPos);
        outValues[insertOutIdx] = val;
        outIndices[insertOutIdx] = colU32;
      }
    }
  }
}
