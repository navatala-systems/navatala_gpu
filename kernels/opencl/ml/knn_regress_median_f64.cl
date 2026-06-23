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
__kernel void navatala_ml_knn_regress_median_f64(__global const uint* neighborIndices, __global const double* targetValues, __global const uint* q, __global const uint* k, __global double* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    __local double sortBuf[256];
    uint baseIdx = (gid * kVal);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      double targetVal = targetValues[neighborIdx];
      sortBuf[jU32] = targetVal;
    }
    uint jIdxAccum = (uint)(0u);
    for (int i = 0; i < (int)(kVal); ++i) {
      uint iU32 = ((uint)(i));
      uint zeroU32 = ((uint)(0));
      if (iU32 > zeroU32) {
        double key = sortBuf[iU32];
        uint initialJIdx = (iU32 - (uint)(1u));
        jIdxAccum = initialJIdx;
        for (int iter = 0; iter < (int)(iU32); ++iter) {
          uint currJIdx = jIdxAccum;
          double currentVal = sortBuf[currJIdx];
          bool shouldShift = (currentVal > key);
          if (shouldShift) {
            uint jPlus1 = (currJIdx + (uint)(1u));
            sortBuf[jPlus1] = currentVal;
            uint newJIdx = (((currJIdx == (uint)(0u))) ? ((uint)(0u)) : ((currJIdx - (uint)(1u))));
            jIdxAccum = newJIdx;
          }
        }
        uint finalJIdx = jIdxAccum;
        double lastCurrentVal = sortBuf[finalJIdx];
        uint insertPos = (((lastCurrentVal > key)) ? (finalJIdx) : ((finalJIdx + (uint)(1u))));
        sortBuf[insertPos] = key;
      }
    }
    uint midIdx = (kVal / (uint)(2u));
    uint kMod2 = (kVal % (uint)(2u));
    bool isOdd = (kMod2 != (uint)(0u));
    double midVal = sortBuf[midIdx];
    uint midMinus1Idx = (((midIdx == (uint)(0u))) ? ((uint)(0u)) : ((midIdx - (uint)(1u))));
    double midMinus1Val = sortBuf[midMinus1Idx];
    double evenMedian = ((midMinus1Val + midVal) / as_double(0x4000000000000000ul));
    double median = ((isOdd) ? (midVal) : (evenMedian));
    predictions[gid] = median;
  }
}
