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
extern "C" __global__ void navatala_dataframe_radix_scatter_f32(const float* _input, const unsigned int* count, const unsigned int* bitOffset, const unsigned int* prefixSums, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int bits = bitOffset[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float val = _input[gid];
    float scaled = (val * __uint_as_float(0x4b800000u));
    int sortKeyRaw = ((int)(scaled));
    unsigned int sortKey = ((unsigned int)((sortKeyRaw + 2147483648)));
    unsigned int digit = ((sortKey >> bits) & 255u);
    unsigned int destBase = prefixSums[digit];
    _output[destBase] = val;
  }
}
