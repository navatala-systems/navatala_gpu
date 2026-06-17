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
extern "C" __global__ void navatala_dataframe_add_f64(const double* left, const unsigned int* leftValid, const double* right, const unsigned int* rightValid, const unsigned int* count, double* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    double lval = left[gid];
    double rval = right[gid];
    double result = (lval + rval);
    _output[gid] = result;
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int leftValidWord = leftValid[wordIdx];
    unsigned int leftValidBit = ((leftValidWord >> bitIdx) & 1u);
    unsigned int rightValidWord = rightValid[wordIdx];
    unsigned int rightValidBit = ((rightValidWord >> bitIdx) & 1u);
    unsigned int outValidBit = (leftValidBit & rightValidBit);
    unsigned int outBit = (outValidBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
