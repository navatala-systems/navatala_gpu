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
extern "C" __global__ void navatala_dataframe_group_by_nunique_helper_i64(const long long* data, const unsigned int* validity, const unsigned int* groupIds, const unsigned int* count, unsigned int* hashOutput, unsigned int* validOutput) {
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
    long long val = data[gid];
    unsigned int groupId = groupIds[gid];
    unsigned int kLow = ((unsigned int)(val));
    unsigned int kHigh = ((unsigned int)((val >> 32)));
    unsigned int c1 = 3432918353u;
    unsigned int c2 = 461845907u;
    unsigned int k1a = (kLow * c1);
    unsigned int k1b = ((k1a << 15u) | (k1a >> 17u));
    unsigned int k1c = (k1b * c2);
    unsigned int k2a = (kHigh * c1);
    unsigned int k2b = ((k2a << 15u) | (k2a >> 17u));
    unsigned int k2c = (k2b * c2);
    unsigned int h1 = (groupId ^ k1c);
    unsigned int h2 = (h1 ^ k2c);
    unsigned int h3 = ((h2 << 13u) | (h2 >> 19u));
    unsigned int h4 = ((h3 * 5u) + 3864292196u);
    unsigned int h5 = (h4 ^ 8u);
    unsigned int h6 = (h5 ^ (h5 >> 16u));
    unsigned int h7 = (h6 * 2246822507u);
    unsigned int h8 = (h7 ^ (h7 >> 13u));
    unsigned int h9 = (h8 * 3266489909u);
    unsigned int hash = (h9 ^ (h9 >> 16u));
    unsigned int outHash = ((isValid) ? (hash) : (0u));
    hashOutput[gid] = outHash;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&validOutput[wordIdx], outBit);
  }
}
