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
extern "C" __global__ void navatala_cfd_primitives_div_face_flux_csr_f64(const int* offsets, const int* faceIdx, const float* sign, const double* faceFlux, const double* vol, const int* params, double* outDiv) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double term = (signV * faceFlux[f]);
      acc = (acc + term);
    }
    outDiv[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (acc / ((double)(vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])));
  }
}
