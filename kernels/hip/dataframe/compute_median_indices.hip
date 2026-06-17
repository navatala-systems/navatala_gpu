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
extern "C" __global__ void navatala_dataframe_compute_median_indices(const unsigned int* count, unsigned int* floorIdx, unsigned int* ceilIdx, unsigned int* isEven) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int n = count[0u];
  unsigned int mod2 = (n % 2u);
  bool even = (mod2 == 0u);
  unsigned int _half = (n / 2u);
  unsigned int floor = ((even) ? ((_half - 1u)) : (_half));
  unsigned int ceil = _half;
  floorIdx[0u] = floor;
  ceilIdx[0u] = ceil;
  unsigned int evenFlag = ((even) ? (1u) : (0u));
  isEven[0u] = evenFlag;
}
