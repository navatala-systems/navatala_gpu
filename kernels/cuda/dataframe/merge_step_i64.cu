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
extern "C" __global__ void navatala_dataframe_merge_step_i64(const long long* _input, const unsigned int* count, const unsigned int* blockSize, long long* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int bSize = blockSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int mergeBlockSize = (bSize * 2u);
    unsigned int mergeBlockIdx = (gid / mergeBlockSize);
    unsigned int posInMerge = (gid % mergeBlockSize);
    unsigned int leftStart = (mergeBlockIdx * mergeBlockSize);
    unsigned int rightStart = (leftStart + bSize);
    unsigned int leftEnd = (((rightStart < n)) ? (rightStart) : (n));
    unsigned int rightEnd = ((((leftStart + mergeBlockSize) < n)) ? ((leftStart + mergeBlockSize)) : (n));
    unsigned int leftSize = (leftEnd - leftStart);
    unsigned int rightSize = (rightEnd - rightStart);
    bool leftDone = (posInMerge >= leftSize);
    unsigned int leftIdx = (((posInMerge < leftSize)) ? (posInMerge) : ((leftSize - 1u)));
    unsigned int rightIdx = ((leftDone) ? ((posInMerge - leftSize)) : (0u));
    unsigned int absLeftIdx = (leftStart + leftIdx);
    unsigned int absRightIdx = (rightStart + rightIdx);
    long long leftVal = _input[absLeftIdx];
    long long rightVal = (((absRightIdx < rightEnd)) ? (_input[absRightIdx]) : (9223372036854775807));
    bool rightExhausted = (absRightIdx >= rightEnd);
    bool leftSmaller = (leftVal <= rightVal);
    bool takeLeft = ((!leftDone) && (rightExhausted || leftSmaller));
    long long outVal = ((takeLeft) ? (leftVal) : (rightVal));
    _output[gid] = outVal;
  }
}
