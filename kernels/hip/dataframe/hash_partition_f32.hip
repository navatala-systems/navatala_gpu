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
extern "C" __global__ void navatala_dataframe_hash_partition_f32(const float* keys, const unsigned int* numPartitions, const unsigned int* count, unsigned int* partitionIds) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int numParts = numPartitions[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float key = keys[gid];
    float scaled = (key * __uint_as_float(0x4f000000u));
    int k = ((int)(scaled));
    unsigned int kU = ((unsigned int)(k));
    unsigned int h1 = (kU ^ (kU >> 16u));
    unsigned int h2 = (h1 * 2246822507u);
    unsigned int h3 = (h2 ^ (h2 >> 13u));
    unsigned int h4 = (h3 * 3266489909u);
    unsigned int hash = (h4 ^ (h4 >> 16u));
    unsigned int partId = (hash % numParts);
    partitionIds[gid] = partId;
  }
}
