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
extern "C" __global__ void navatala_dataframe_flatten_offsets(const int* outerOffsets, const int* innerOffsets, const unsigned int* count, int* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int outerIdx = outerOffsets[gid];
    bool isNegative = (outerIdx < 0);
    unsigned int safeIdx = ((isNegative) ? (0u) : (((unsigned int)(outerIdx))));
    int flatOffset = innerOffsets[safeIdx];
    int result = ((isNegative) ? (-1) : (flatOffset));
    _output[gid] = result;
  }
}
