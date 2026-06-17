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
extern "C" __global__ void navatala_dataframe_nested_loop_join_gather_i32(const int* leftData, const int* rightData, const int* leftIndices, const int* rightIndices, const unsigned int* count, int* leftOutput, int* rightOutput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int leftIdx = leftIndices[gid];
    int rightIdx = rightIndices[gid];
    unsigned int leftIdxU = ((unsigned int)(leftIdx));
    unsigned int rightIdxU = ((unsigned int)(rightIdx));
    int leftVal = leftData[leftIdxU];
    int rightVal = rightData[rightIdxU];
    leftOutput[gid] = leftVal;
    rightOutput[gid] = rightVal;
  }
}
