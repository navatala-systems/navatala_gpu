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
extern "C" __global__ void navatala_ml_soft_threshold_f64(const double* value, const double* threshold, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  bool inBounds = (gid == 0u);
  if (inBounds) {
    double v = value[0];
    double t = threshold[0];
    double absV = abs(v);
    double diff = (absV - t);
    double maxDiff = (((diff > __longlong_as_double(0x0000000000000000ull))) ? (diff) : (__longlong_as_double(0x0000000000000000ull)));
    bool positive = (v > __longlong_as_double(0x0000000000000000ull));
    bool negative = (v < __longlong_as_double(0x0000000000000000ull));
    double signVal = ((positive) ? (__longlong_as_double(0x3ff0000000000000ull)) : (((negative) ? (__longlong_as_double(0xbff0000000000000ull)) : (__longlong_as_double(0x0000000000000000ull)))));
    double res = (signVal * maxDiff);
    result[0] = res;
  }
}
