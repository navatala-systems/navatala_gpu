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
extern "C" __global__ void navatala_dataframe_arg_min_f32(const float* _input, const unsigned int* count, const unsigned int* tiePolicy, float* minValue, unsigned int* minIndex) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float svals[256];
  __shared__ unsigned int sidxs[256];
  float otherVal = __uint_as_float(0x00000000u);
  unsigned int otherIdx = 0u;
  float myVal = __uint_as_float(0x00000000u);
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
  float val = _input[gid];
  if (inBounds) {
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = __uint_as_float(0x7f7fffffu);
    sidxs[lid] = 0u;
  }
  __syncthreads();
  unsigned int aminf32_reductionStride = 128u;
  for (int aminf32_reductionStep = 0; aminf32_reductionStep < (int)(8); ++aminf32_reductionStep) {
    unsigned int aminf32_stride = aminf32_reductionStride;
    if (lid < aminf32_stride) {
      otherVal = svals[(lid + aminf32_stride)];
      otherIdx = sidxs[(lid + aminf32_stride)];
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
    unsigned int aminf32_strideToHalve = aminf32_reductionStride;
    unsigned int aminf32_nextStride = (aminf32_strideToHalve >> 1u);
    aminf32_reductionStride = aminf32_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    minValue[0] = svals[0];
    minIndex[0] = sidxs[0];
  }
}
