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
extern "C" __global__ void navatala_cfd_fvm_su_sp(const float* sp, const float* psi, const float* vol, const int* counts, float* outDiag, float* outSource) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float psiVal = psi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float volVal = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float spVol = (spVal * volVal);
    if (spVal >= __uint_as_float(0x00000000u)) {
      float prevDiag = outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float newDiag = (prevDiag + spVol);
      outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newDiag;
    } else {
      float prevSrc = outSource[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float contrib = (spVol * psiVal);
      float newSrc = (prevSrc - contrib);
      outSource[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newSrc;
    }
  }
}
