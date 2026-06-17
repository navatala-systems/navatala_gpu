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
extern "C" __global__ void navatala_dataframe_extract_skewness_f64(const unsigned int* n, const double* M2, const double* M3, double* skewness) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    unsigned int nVal = n[0];
    double m2Val = M2[0];
    double m3Val = M3[0];
    double nf = ((double)(nVal));
    double sqrtN = sqrt(nf);
    double sqrtM2 = sqrt(m2Val);
    double m2Pow15 = (m2Val * sqrtM2);
    double denomSafe = (((m2Val == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (m2Pow15));
    double numerator = (sqrtN * m3Val);
    double skewRaw = (numerator / denomSafe);
    double skewResult = (((m2Val == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (skewRaw));
    skewness[0] = skewResult;
  }
}
