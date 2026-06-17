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
extern "C" __global__ void navatala_cfd_stf_all_faces_f64(const double* divN, const float* alpha, const float* alphaF, const float* delta, const int* owner, const int* nei, const float* weights, const double* contactGrad, const unsigned int* thetaMask, const float* paramsF, const int* counts, double* stf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    double w = ((double)(weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    double wm = (__longlong_as_double(0x3ff0000000000000ull) - w);
    double divInterp = ((w * divN[o]) + (wm * divN[n]));
    double Kf = (-divInterp);
    double snGrad = __longlong_as_double(0x0000000000000000ull);
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2])) {
      snGrad = (((double)(delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) * (((double)(alpha[n])) - ((double)(alpha[o]))));
    } else {
      if ((thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u)) {
        snGrad = contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      } else {
        snGrad = (((double)(delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) * (((double)(alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) - ((double)(alpha[o]))));
      }
    }
    stf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((((double)(paramsF[0])) * Kf) * snGrad);
  }
}
