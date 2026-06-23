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
extern "C" __global__ void navatala_cfd_matrix_relax(const float* diagOld, const float* sumOff, const float* psi, const int* counts, const float* paramsF, float* diag, float* source) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    float d0 = diagOld[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float sOff = sumOff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float x = psi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float negD0 = (__uint_as_float(0x00000000u) - d0);
    float absD0 = (((d0 >= __uint_as_float(0x00000000u))) ? (d0) : (negD0));
    float maxVal = (((absD0 >= sOff)) ? (absD0) : (sOff));
    float newDiag = (maxVal / paramsF[0]);
    float srcPrev = source[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float delta = (newDiag - d0);
    float srcNew = (srcPrev + (delta * x));
    diag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newDiag;
    source[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = srcNew;
  }
}
