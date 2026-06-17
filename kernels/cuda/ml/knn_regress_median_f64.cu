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
extern "C" __global__ void navatala_ml_knn_regress_median_f64(const unsigned int* neighborIndices, const double* targetValues, const unsigned int* q, const unsigned int* k, double* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int qVal = q[0];
  unsigned int kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    __shared__ double sortBuf[256];
    unsigned int baseIdx = (gid * kVal);
    for (int j = 0; j < (int)(kVal); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int neighborOffset = (baseIdx + jU32);
      unsigned int neighborIdx = neighborIndices[neighborOffset];
      double targetVal = targetValues[neighborIdx];
      sortBuf[jU32] = targetVal;
    }
    unsigned int jIdxAccum = 0u;
    for (int i = 0; i < (int)(kVal); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int zeroU32 = ((unsigned int)(0));
      if ((iU32 > zeroU32)) {
        double key = sortBuf[iU32];
        unsigned int initialJIdx = (iU32 - 1u);
        jIdxAccum = initialJIdx;
        for (int iter = 0; iter < (int)(iU32); ++iter) {
          unsigned int currJIdx = jIdxAccum;
          double currentVal = sortBuf[currJIdx];
          bool shouldShift = (currentVal > key);
          if (shouldShift) {
            unsigned int jPlus1 = (currJIdx + 1u);
            sortBuf[jPlus1] = currentVal;
            unsigned int newJIdx = (((currJIdx == 0u)) ? (0u) : ((currJIdx - 1u)));
            jIdxAccum = newJIdx;
          }
        }
        unsigned int finalJIdx = jIdxAccum;
        double lastCurrentVal = sortBuf[finalJIdx];
        unsigned int insertPos = (((lastCurrentVal > key)) ? (finalJIdx) : ((finalJIdx + 1u)));
        sortBuf[insertPos] = key;
      }
    }
    unsigned int midIdx = (kVal / 2u);
    unsigned int kMod2 = (kVal % 2u);
    bool isOdd = (kMod2 != 0u);
    double midVal = sortBuf[midIdx];
    unsigned int midMinus1Idx = (((midIdx == 0u)) ? (0u) : ((midIdx - 1u)));
    double midMinus1Val = sortBuf[midMinus1Idx];
    double evenMedian = ((midMinus1Val + midVal) / __longlong_as_double(0x4000000000000000ull));
    double median = ((isOdd) ? (midVal) : (evenMedian));
    predictions[gid] = median;
  }
}
