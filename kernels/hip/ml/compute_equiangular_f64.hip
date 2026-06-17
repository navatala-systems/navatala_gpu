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
extern "C" __global__ void navatala_ml_compute_equiangular_f64(const double* correlations, const unsigned int* activeMask, const unsigned int* nCols, double* equiangular) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    unsigned int mask = activeMask[gid];
    bool isActive = (mask == 1u);
    if (isActive) {
      double corr = correlations[gid];
      bool positive = (corr > __longlong_as_double(0x0000000000000000ull));
      bool negative = (corr < __longlong_as_double(0x0000000000000000ull));
      double signVal = ((positive) ? (__longlong_as_double(0x3ff0000000000000ull)) : (((negative) ? (__longlong_as_double(0xbff0000000000000ull)) : (__longlong_as_double(0x0000000000000000ull)))));
      equiangular[gid] = signVal;
    } else {
      equiangular[gid] = __longlong_as_double(0x0000000000000000ull);
    }
  }
}
