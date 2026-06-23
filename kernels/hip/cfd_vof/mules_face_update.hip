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
extern "C" __global__ void navatala_cfd_vof_mules_face_update(const float* phiCorr, float* lambda, const int* owner, const int* nei, const float* lambdam, const float* lambdap, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2])) {
    return;
  } else {
    float pc = phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float a0 = lambdam[o];
    float b0 = lambdap[n];
    float lim = a0;
    if ((lim > b0)) {
      lim = b0;
    }
    if ((pc > __uint_as_float(0x00000000u))) {
      float a1 = lambdap[o];
      float b1 = lambdam[n];
      float lim1 = a1;
      if ((lim1 > b1)) {
        lim1 = b1;
      }
      lim = lim1;
    }
    float cur = lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float _out = cur;
    if ((_out > lim)) {
      _out = lim;
    }
    lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = _out;
  }
}
