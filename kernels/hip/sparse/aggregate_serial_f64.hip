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
extern "C" __global__ void navatala_sparse_aggregate_serial_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const unsigned int* strongMask, const unsigned int* nRows, int* aggregateId, unsigned int* nAggregates) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  int nextAgg = 0;
  for (int i = 0; i < (int)(N); ++i) {
    int rs = ((int)(rowPtr[i]));
    int re = ((int)(rowPtr[(i + 1)]));
    int bestCol = -1;
    double bestVal = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      unsigned int isStrong = strongMask[k];
      if ((isStrong == 1u)) {
        double a = values[k];
        if ((abs(a) > bestVal)) {
          bestVal = abs(a);
          bestCol = ((int)(colIdx[k]));
        }
      }
    }
    if ((bestCol >= 0)) {
      aggregateId[i] = (((i < bestCol)) ? (i) : (bestCol));
    } else {
      aggregateId[i] = nextAgg;
      nextAgg = (nextAgg + 1);
    }
  }
  nAggregates[0] = ((unsigned int)(nextAgg));
}
