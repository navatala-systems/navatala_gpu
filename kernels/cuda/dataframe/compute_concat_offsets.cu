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
extern "C" __global__ void navatala_dataframe_compute_concat_offsets(const unsigned int* lengths, const unsigned int* numColumns, unsigned int* offsets) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = numColumns[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    bool isFirst = (gid == 0u);
    unsigned int len0 = (((n >= 1u)) ? (lengths[0u]) : (0u));
    unsigned int len1 = (((n >= 2u)) ? (lengths[1u]) : (0u));
    unsigned int len2 = (((n >= 3u)) ? (lengths[2u]) : (0u));
    unsigned int len3 = (((n >= 4u)) ? (lengths[3u]) : (0u));
    unsigned int prefix0 = 0u;
    unsigned int prefix1 = len0;
    unsigned int prefix2 = (prefix1 + len1);
    unsigned int prefix3 = (prefix2 + len2);
    bool isIdx0 = (gid == 0u);
    bool isIdx1 = (gid == 1u);
    bool isIdx2 = (gid == 2u);
    bool isIdx3 = (gid == 3u);
    unsigned int result = ((isIdx0) ? (prefix0) : (((isIdx1) ? (prefix1) : (((isIdx2) ? (prefix2) : (((isIdx3) ? (prefix3) : (0u))))))));
    offsets[gid] = result;
  }
}
