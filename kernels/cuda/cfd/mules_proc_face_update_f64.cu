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
extern "C" __global__ void navatala_cfd_mules_proc_face_update_f64(const int* procFaces, const int* counts, const double* phiCorr, double* lambda, const double* lambdapCell, const double* lambdamCell, const int* ownerAll, const double* nbrLambdaP, const double* nbrLambdaM) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int face = procFaces[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = ownerAll[face];
    double pc = phiCorr[face];
    double lim = lambdamCell[o];
    double other = nbrLambdaP[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (lim > other) {
      lim = other;
    }
    if (pc > __longlong_as_double(0x0000000000000000ull)) {
      double limPos = lambdapCell[o];
      double otherPos = nbrLambdaM[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (limPos > otherPos) {
        limPos = otherPos;
      }
      lim = limPos;
    }
    double cur = lambda[face];
    double _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[face] = _out;
  }
}
