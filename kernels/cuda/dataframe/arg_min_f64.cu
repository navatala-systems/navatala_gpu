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
extern "C" __global__ void navatala_dataframe_arg_min_f64(const double* _input, const unsigned int* count, const unsigned int* tiePolicy, double* minValue, unsigned int* minIndex) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double svals[256];
  __shared__ unsigned int sidxs[256];
  double otherVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int otherIdx = 0u;
  double myVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int myIdx = 0u;
  bool otherLess = false;
  bool equalVals = false;
  bool otherIdxSmaller = false;
  bool takeOtherFirst = false;
  bool takeOther = false;
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  unsigned int policy = tiePolicy[0];
  bool useFirst = (policy == 0u);
  double val = _input[gid];
  if (inBounds) {
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = __longlong_as_double(0x7fefffffffffffffull);
    sidxs[lid] = 0u;
  }
  __syncthreads();
  unsigned int aminf64_reductionStride = 128u;
  for (int aminf64_reductionStep = 0; aminf64_reductionStep < (int)(8); ++aminf64_reductionStep) {
    unsigned int aminf64_stride = aminf64_reductionStride;
    if ((lid < aminf64_stride)) {
      otherVal = svals[(lid + aminf64_stride)];
      otherIdx = sidxs[(lid + aminf64_stride)];
      myVal = svals[lid];
      myIdx = sidxs[lid];
      otherLess = (otherVal < myVal);
      equalVals = (otherVal == myVal);
      otherIdxSmaller = (otherIdx < myIdx);
      takeOtherFirst = (otherLess || (equalVals && otherIdxSmaller));
      takeOther = ((useFirst) ? (takeOtherFirst) : (otherLess));
      if (takeOther) {
        svals[lid] = otherVal;
        sidxs[lid] = otherIdx;
      }
    }
    unsigned int aminf64_strideToHalve = aminf64_reductionStride;
    unsigned int aminf64_nextStride = (aminf64_strideToHalve >> 1u);
    aminf64_reductionStride = aminf64_nextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    minValue[0] = svals[0];
    minIndex[0] = sidxs[0];
  }
}
