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
extern "C" __global__ void navatala_dataframe_cross_join_gather_i32(const int* leftData, const int* rightData, const unsigned int* leftCount, const unsigned int* rightCount, const unsigned long long* outputCount, int* leftOutput, int* rightOutput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned long long gid = ((unsigned long long)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned long long n = outputCount[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int rCount = rightCount[0u];
    unsigned long long rCount64 = ((unsigned long long)(rCount));
    unsigned long long leftIdx64 = (gid / rCount64);
    unsigned long long rightIdx64 = (gid % rCount64);
    unsigned int leftIdx = ((unsigned int)(leftIdx64));
    unsigned int rightIdx = ((unsigned int)(rightIdx64));
    int leftVal = leftData[leftIdx];
    int rightVal = rightData[rightIdx];
    unsigned int outIdx = ((unsigned int)(gid));
    leftOutput[outIdx] = leftVal;
    rightOutput[outIdx] = rightVal;
  }
}
