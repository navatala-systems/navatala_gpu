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
extern "C" __global__ void navatala_dataframe_reservoir_update256_f64(const double* x, double* reservoir, unsigned int* count, unsigned int* seed) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    double xVal = x[0];
    unsigned int n = count[0];
    unsigned int s = seed[0];
    unsigned int s1 = (s ^ (s << 13u));
    unsigned int s2 = (s1 ^ (s1 >> 17u));
    unsigned int s3 = (s2 ^ (s2 << 5u));
    unsigned int nPlus1 = (n + 1u);
    unsigned int j = (s3 % nPlus1);
    bool fillPhase = (n < 256u);
    bool replacePhase = ((n >= 256u) && (j < 256u));
    if (fillPhase) {
      reservoir[n] = xVal;
    }
    if (replacePhase) {
      reservoir[j] = xVal;
    }
    count[0] = nPlus1;
    seed[0] = s3;
  }
}
