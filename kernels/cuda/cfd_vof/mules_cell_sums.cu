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
extern "C" __global__ void navatala_cfd_vof_mules_cell_sums(const float* phiCorr, const float* lambda, const int* offsets, const int* faceIdx, const float* sign, const int* counts, const float* paramsF, float* sumlPhip, float* mSumlPhim) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sp = __uint_as_float(0x00000000u);
    float sm = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float lpc = (lambda[f] * phiCorr[f]);
      float s = sign[k];
      if (f < counts[2]) {
        if (s > __uint_as_float(0x00000000u)) {
          if (lpc > __uint_as_float(0x00000000u)) {
            sp = (sp + lpc);
          } else {
            sm = (sm + (-lpc));
          }
        } else {
          if (lpc > __uint_as_float(0x00000000u)) {
            sm = (sm + lpc);
          } else {
            sp = (sp + (-lpc));
          }
        }
      } else {
        if (lpc > __uint_as_float(0x00000000u)) {
          sp = (sp + lpc);
        } else {
          sm = (sm + (-lpc));
        }
      }
    }
    sumlPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sp;
    mSumlPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sm;
  }
}
