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
extern "C" __global__ void navatala_linalg_elt_sign_f64(const double* A, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((i < n[0u])) {
    double a = A[i];
    bool isNeg = (a < __longlong_as_double(0x0000000000000000ull));
    double negOrZero = ((isNeg) ? (__longlong_as_double(0xbff0000000000000ull)) : (__longlong_as_double(0x0000000000000000ull)));
    bool isPos = (a > __longlong_as_double(0x0000000000000000ull));
    double result = ((isPos) ? (__longlong_as_double(0x3ff0000000000000ull)) : (negOrZero));
    C[i] = result;
  }
}
