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
extern "C" __global__ void navatala_dataframe_tdigest_cdf_f64(const double* x, const double* centroidMeans, const double* centroidWeights, const unsigned int* centroidCount, const double* totalWeight, const double* minVal, const double* maxVal, double* cdf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    double xVal = x[0];
    unsigned int count = centroidCount[0];
    double minV = minVal[0];
    double maxV = maxVal[0];
    double range = (maxV - minV);
    double safeRange = (((range == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (range));
    double rawCdf = ((xVal - minV) / safeRange);
    double clampLo = (((rawCdf < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (rawCdf));
    double clampHi = (((clampLo > __longlong_as_double(0x3ff0000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (clampLo));
    bool isEmpty = (count == 0u);
    double finalCdf = ((isEmpty) ? (__longlong_as_double(0x0000000000000000ull)) : (clampHi));
    cdf[0] = finalCdf;
  }
}
