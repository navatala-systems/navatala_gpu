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
extern "C" __global__ void navatala_dataframe_combine_hash_values(const unsigned int* hashes1, const unsigned int* hashes2, const unsigned int* count, unsigned int* combined) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int h1 = hashes1[gid];
    unsigned int h2 = hashes2[gid];
    unsigned int golden = 2654435769u;
    unsigned int h2Left = (h2 << 5u);
    unsigned int h2Right = (h2 >> 27u);
    unsigned int h2Rotated = (h2Left | h2Right);
    unsigned int h2WithGolden = (h2Rotated + golden);
    unsigned int result = (h1 ^ h2WithGolden);
    combined[gid] = result;
  }
}
