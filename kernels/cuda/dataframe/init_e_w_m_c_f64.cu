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
extern "C" __global__ void navatala_dataframe_init_e_w_m_c_f64(const double* alphaIn, double* alpha, double* nEff, double* meanX, double* meanY, double* Cxy, double* varX, double* varY) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    double a = alphaIn[0];
    alpha[0] = a;
    nEff[0] = __longlong_as_double(0x0000000000000000ull);
    meanX[0] = __longlong_as_double(0x0000000000000000ull);
    meanY[0] = __longlong_as_double(0x0000000000000000ull);
    Cxy[0] = __longlong_as_double(0x0000000000000000ull);
    varX[0] = __longlong_as_double(0x0000000000000000ull);
    varY[0] = __longlong_as_double(0x0000000000000000ull);
  }
}
