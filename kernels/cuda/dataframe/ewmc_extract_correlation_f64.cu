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
extern "C" __global__ void navatala_dataframe_ewmc_extract_correlation_f64(const double* Cxy, const double* varX, const double* varY, const double* nEff, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    double c = Cxy[0];
    double vx = varX[0];
    double vy = varY[0];
    double n = nEff[0];
    double vxvy = (vx * vy);
    double eps = __longlong_as_double(0x3cd203af9ee75616ull);
    double denom = sqrt(vxvy);
    double safeDenom = (((vxvy < eps)) ? (__longlong_as_double(0x3ff0000000000000ull)) : (denom));
    double corr = (c / safeDenom);
    double res = (((vxvy < eps)) ? (__longlong_as_double(0x0000000000000000ull)) : (corr));
    result[0] = res;
  }
}
