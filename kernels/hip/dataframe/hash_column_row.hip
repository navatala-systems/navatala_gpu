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
extern "C" __global__ void navatala_dataframe_hash_column_row(const unsigned int* colHashes1, const unsigned int* colHashes2, const unsigned int* count, unsigned int* rowHashes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int h1 = colHashes1[gid];
    unsigned int h2 = colHashes2[gid];
    unsigned int c1 = 3432918353u;
    unsigned int c2 = 461845907u;
    unsigned int h1Left = (h1 << 13u);
    unsigned int h1Right = (h1 >> 19u);
    unsigned int h1Rotated = (h1Left | h1Right);
    unsigned int combined1 = (h1Rotated ^ h2);
    unsigned int mixed1 = (combined1 * c1);
    unsigned int m1Left = (mixed1 << 15u);
    unsigned int m1Right = (mixed1 >> 17u);
    unsigned int m1Rotated = (m1Left | m1Right);
    unsigned int mixed2 = (m1Rotated * c2);
    unsigned int finalMix = (mixed2 ^ (mixed2 >> 16u));
    rowHashes[gid] = finalMix;
  }
}
