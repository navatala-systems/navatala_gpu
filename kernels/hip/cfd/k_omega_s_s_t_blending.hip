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
extern "C" __global__ void navatala_cfd_k_omega_s_s_t_blending(const float* k, const float* omega, const float* nut, const float* nu, const float* y, const float* gradKx, const float* gradKy, const float* gradKz, const float* gradOmegax, const float* gradOmegay, const float* gradOmegaz, const int* counts, float* outF1, float* outF23, float* outCDkOmega) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    float kC = k[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float omC = omega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nutC = nut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nuC = nu[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float yC = y[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gKx = gradKx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gKy = gradKy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gKz = gradKz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gOx = gradOmegax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gOy = gradOmegay[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gOz = gradOmegaz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float dotGrad = (((gKx * gOx) + (gKy * gOy)) + (gKz * gOz));
    float omSafe = (((omC > __uint_as_float(0x2edbe6ffu))) ? (omC) : (__uint_as_float(0x2edbe6ffu)));
    float cdkw = ((__uint_as_float(0x3fdb22d1u) * dotGrad) / omSafe);
    float cdkwPlus = (((cdkw > __uint_as_float(0x2edbe6ffu))) ? (cdkw) : (__uint_as_float(0x2edbe6ffu)));
    float ySq = (yC * yC);
    float ySqSafe = (((ySq > __uint_as_float(0x1e3ce508u))) ? (ySq) : (__uint_as_float(0x1e3ce508u)));
    float kSafe = (((kC > __uint_as_float(0x00000000u))) ? (kC) : (__uint_as_float(0x00000000u)));
    float sqrtK = sqrt(kSafe);
    float denom1 = (__uint_as_float(0x3db851ecu) * (omSafe * yC));
    float denom1Safe = (((denom1 > __uint_as_float(0x2edbe6ffu))) ? (denom1) : (__uint_as_float(0x2edbe6ffu)));
    float term1 = (sqrtK / denom1Safe);
    float denom2 = (ySqSafe * omSafe);
    float term2 = ((__uint_as_float(0x43fa0000u) * nuC) / denom2);
    float maxT12 = (((term1 >= term2)) ? (term1) : (term2));
    float denom3 = (cdkwPlus * ySqSafe);
    float denom3Safe = (((denom3 > __uint_as_float(0x2edbe6ffu))) ? (denom3) : (__uint_as_float(0x2edbe6ffu)));
    float term3 = ((__uint_as_float(0x405b22d1u) * kC) / denom3Safe);
    float minT3 = (((maxT12 <= term3)) ? (maxT12) : (term3));
    float arg1 = (((minT3 <= __uint_as_float(0x41200000u))) ? (minT3) : (__uint_as_float(0x41200000u)));
    float arg1sq = (arg1 * arg1);
    float arg1p4 = (arg1sq * arg1sq);
    float arg1p4Safe = (((arg1p4 <= __uint_as_float(0x42200000u))) ? (arg1p4) : (__uint_as_float(0x42200000u)));
    float f1Val = tanh(arg1p4Safe);
    float term2a = ((__uint_as_float(0x40000000u) * sqrtK) / denom1Safe);
    float maxF2 = (((term2a >= term2)) ? (term2a) : (term2));
    float arg2 = (((maxF2 <= __uint_as_float(0x42c80000u))) ? (maxF2) : (__uint_as_float(0x42c80000u)));
    float arg2sq = (arg2 * arg2);
    float arg2sqSafe = (((arg2sq <= __uint_as_float(0x42200000u))) ? (arg2sq) : (__uint_as_float(0x42200000u)));
    float f23Val = tanh(arg2sqSafe);
    outF1[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = f1Val;
    outF23[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = f23Val;
    outCDkOmega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = cdkw;
  }
}
