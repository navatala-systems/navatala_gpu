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
extern "C" __global__ void navatala_dataframe_hash_keys_i32(const int* keys, const unsigned int* validity, const unsigned int* count, const unsigned int* tableSize, unsigned int* hashes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int tSize = tableSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int key = keys[gid];
    unsigned int k1 = ((unsigned int)(key));
    unsigned int k2 = (k1 * 3432918353u);
    unsigned int k3 = ((k2 << 15u) | (k2 >> 17u));
    unsigned int k4 = (k3 * 461845907u);
    unsigned int h1 = (k4 ^ 2246822507u);
    unsigned int h2 = (h1 ^ (h1 >> 16u));
    unsigned int h3 = (h2 * 3266489909u);
    unsigned int h4 = (h3 ^ (h3 >> 13u));
    unsigned int hashIdx = (h4 % tSize);
    unsigned int outHash = ((isValid) ? (hashIdx) : (2147483647u));
    hashes[gid] = outHash;
  }
}
