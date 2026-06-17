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
extern "C" __global__ void navatala_dataframe_compute_quantile_indices(const unsigned int* quantile, const unsigned int* count, unsigned int* floorIdx, unsigned int* ceilIdx, unsigned int* fraction) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int q = quantile[0u];
  unsigned int n = count[0u];
  unsigned int nMinus1 = (n - 1u);
  unsigned int posScaled = (q * nMinus1);
  unsigned int floor = (posScaled / 1000u);
  unsigned int ceilRaw = (floor + 1u);
  unsigned int ceil = (((ceilRaw < n)) ? (ceilRaw) : (nMinus1));
  unsigned int frac = (posScaled % 1000u);
  floorIdx[0u] = floor;
  ceilIdx[0u] = ceil;
  fraction[0u] = frac;
}
