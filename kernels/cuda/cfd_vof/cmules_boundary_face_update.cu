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
extern "C" __global__ void navatala_cfd_vof_cmules_boundary_face_update(const float* phi, const float* phiCorr, float* lambda, const int* ownerAll, const float* lambdam, const float* lambdap, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2]) {
      return;
    } else {
      float pc = phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float combined = (phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + pc);
      float eps2 = (paramsF[1] * paramsF[1]);
      if (combined <= eps2) {
        return;
      } else {
        int o = ownerAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float lim = lambdam[o];
        if (pc > __uint_as_float(0x00000000u)) {
          lim = lambdap[o];
        }
        float cur = lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float _out = cur;
        if (_out > lim) {
          _out = lim;
        }
        lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = _out;
      }
    }
  }
}
