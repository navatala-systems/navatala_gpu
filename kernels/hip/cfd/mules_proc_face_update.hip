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
extern "C" __global__ void navatala_cfd_mules_proc_face_update(const int* procFaces, const int* counts, const float* phiCorr, float* lambda, const float* lambdapCell, const float* lambdamCell, const int* ownerAll, const float* nbrLambdaP, const float* nbrLambdaM) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int face = procFaces[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = ownerAll[face];
    float pc = phiCorr[face];
    float lim = lambdamCell[o];
    float other = nbrLambdaP[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (lim > other) {
      lim = other;
    }
    if (pc > __uint_as_float(0x00000000u)) {
      float limPos = lambdapCell[o];
      float otherPos = nbrLambdaM[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (limPos > otherPos) {
        limPos = otherPos;
      }
      lim = limPos;
    }
    float cur = lambda[face];
    float _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[face] = _out;
  }
}
