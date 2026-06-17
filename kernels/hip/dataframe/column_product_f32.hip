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
extern "C" __global__ void navatala_dataframe_column_product_f32(const float* data, const unsigned int* validity, const unsigned int* count, long long* sumLogResult, unsigned int* negCount, unsigned int* zeroFlag) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    if (isValid) {
      float val = data[gid];
      bool isZero = (val == __uint_as_float(0x00000000u));
      if (isZero) {
        atomicMax(&zeroFlag[0u], 1u);
      } else {
        bool isNeg = (val < __uint_as_float(0x00000000u));
        if (isNeg) {
          atomicAdd(&negCount[0u], 1u);
        }
        float absVal = abs(val);
        float logVal = log(absVal);
        float scaleFactor = __uint_as_float(0x49800000u);
        float scaledLog = (logVal * scaleFactor);
        long long scaledLogInt = ((long long)(scaledLog));
        atomicAdd(&sumLogResult[0u], scaledLogInt);
      }
    }
  }
}
