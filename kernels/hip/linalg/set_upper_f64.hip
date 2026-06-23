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
extern "C" __global__ void navatala_linalg_set_upper_f64(const double* val, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  double fillVal = val[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    bool isStrictUpper = (i < j);
    if (isStrictUpper) {
      A[idx] = fillVal;
    }
  }
}
