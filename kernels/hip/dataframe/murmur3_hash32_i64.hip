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
extern "C" __global__ void navatala_dataframe_murmur3_hash32_i64(const long long* keys, const unsigned int* seed, const unsigned int* count, unsigned int* hashes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    long long key = keys[gid];
    unsigned int kLow = ((unsigned int)(key));
    unsigned int kHigh = ((unsigned int)((key >> 32)));
    unsigned int c1 = 3432918353u;
    unsigned int c2 = 461845907u;
    unsigned int k1a = (kLow * c1);
    unsigned int k1b = ((k1a << 15u) | (k1a >> 17u));
    unsigned int k1c = (k1b * c2);
    unsigned int h1a = (s ^ k1c);
    unsigned int h1b = ((h1a << 13u) | (h1a >> 19u));
    unsigned int h1c = ((h1b * 5u) + 3864292196u);
    unsigned int k2a = (kHigh * c1);
    unsigned int k2b = ((k2a << 15u) | (k2a >> 17u));
    unsigned int k2c = (k2b * c2);
    unsigned int h2a = (h1c ^ k2c);
    unsigned int h2b = ((h2a << 13u) | (h2a >> 19u));
    unsigned int h2c = ((h2b * 5u) + 3864292196u);
    unsigned int h3 = (h2c ^ 8u);
    unsigned int h4 = (h3 ^ (h3 >> 16u));
    unsigned int h5 = (h4 * 2246822507u);
    unsigned int h6 = (h5 ^ (h5 >> 13u));
    unsigned int h7 = (h6 * 3266489909u);
    unsigned int hash = (h7 ^ (h7 >> 16u));
    hashes[gid] = hash;
  }
}
