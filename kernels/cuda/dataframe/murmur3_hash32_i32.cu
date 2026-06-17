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
extern "C" __global__ void navatala_dataframe_murmur3_hash32_i32(const int* keys, const unsigned int* seed, const unsigned int* count, unsigned int* hashes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int key = keys[gid];
    unsigned int k = ((unsigned int)(key));
    unsigned int c1 = 3432918353u;
    unsigned int c2 = 461845907u;
    unsigned int k1 = (k * c1);
    unsigned int k2 = ((k1 << 15u) | (k1 >> 17u));
    unsigned int k3 = (k2 * c2);
    unsigned int h1 = (s ^ k3);
    unsigned int h2 = ((h1 << 13u) | (h1 >> 19u));
    unsigned int h3 = ((h2 * 5u) + 3864292196u);
    unsigned int h4 = (h3 ^ 4u);
    unsigned int h5 = (h4 ^ (h4 >> 16u));
    unsigned int h6 = (h5 * 2246822507u);
    unsigned int h7 = (h6 ^ (h6 >> 13u));
    unsigned int h8 = (h7 * 3266489909u);
    unsigned int hash = (h8 ^ (h8 >> 16u));
    hashes[gid] = hash;
  }
}
