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
extern "C" __global__ void navatala_linalg_batched_dot_f32(const float* a, const float* b, const unsigned int* vecLen, const unsigned int* batchSize, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int len = vecLen[0u];
  unsigned int numBatches = batchSize[0u];
  unsigned int k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    unsigned int baseOffset = (k * len);
    float sumAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(len); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = (baseOffset + iU32);
      float aVal = a[idx];
      float bVal = b[idx];
      float prod = (aVal * bVal);
      float currentSum = sumAccum;
      float nextSum = (currentSum + prod);
      sumAccum = nextSum;
    }
    float finalSum = sumAccum;
    result[k] = finalSum;
  }
}
