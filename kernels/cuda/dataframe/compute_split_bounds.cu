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
extern "C" __global__ void navatala_dataframe_compute_split_bounds(const unsigned int* totalRows, const unsigned int* numSplits, int* splitStarts, int* splitEnds) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nRows = totalRows[0u];
  unsigned int nSplits = numSplits[0u];
  bool inBounds = (gid < nSplits);
  if (inBounds) {
    unsigned int baseSize = (nRows / nSplits);
    unsigned int remainder = (nRows % nSplits);
    bool hasExtra = (gid < remainder);
    unsigned int extraRows = ((hasExtra) ? (1u) : (0u));
    unsigned int splitSize = (baseSize + extraRows);
    unsigned int prevSplitsWithExtra = (((gid < remainder)) ? (gid) : (remainder));
    unsigned int prevSplitsWithoutExtra = (gid - prevSplitsWithExtra);
    unsigned int sizePlusOne = (baseSize + 1u);
    unsigned int startFromExtra = (prevSplitsWithExtra * sizePlusOne);
    unsigned int startFromRegular = (prevSplitsWithoutExtra * baseSize);
    unsigned int startIdx = (startFromExtra + startFromRegular);
    unsigned int endIdx = (startIdx + splitSize);
    int startInt = ((int)(startIdx));
    int endInt = ((int)(endIdx));
    splitStarts[gid] = startInt;
    splitEnds[gid] = endInt;
  }
}
