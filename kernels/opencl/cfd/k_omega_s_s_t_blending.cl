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

__kernel void navatala_cfd_k_omega_s_s_t_blending(__global const float* k, __global const float* omega, __global const float* nut, __global const float* nu, __global const float* y, __global const float* gradKx, __global const float* gradKy, __global const float* gradKz, __global const float* gradOmegax, __global const float* gradOmegay, __global const float* gradOmegaz, __global const int* counts, __global float* outF1, __global float* outF23, __global float* outCDkOmega) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float kC = k[((int)((int)(get_global_id(0))))];
    float omC = omega[((int)((int)(get_global_id(0))))];
    float nutC = nut[((int)((int)(get_global_id(0))))];
    float nuC = nu[((int)((int)(get_global_id(0))))];
    float yC = y[((int)((int)(get_global_id(0))))];
    float gKx = gradKx[((int)((int)(get_global_id(0))))];
    float gKy = gradKy[((int)((int)(get_global_id(0))))];
    float gKz = gradKz[((int)((int)(get_global_id(0))))];
    float gOx = gradOmegax[((int)((int)(get_global_id(0))))];
    float gOy = gradOmegay[((int)((int)(get_global_id(0))))];
    float gOz = gradOmegaz[((int)((int)(get_global_id(0))))];
    float dotGrad = (((gKx * gOx) + (gKy * gOy)) + (gKz * gOz));
    float omSafe = (((omC > as_float(0x2edbe6ffu))) ? (omC) : (as_float(0x2edbe6ffu)));
    float cdkw = ((as_float(0x3fdb22d1u) * dotGrad) / omSafe);
    float cdkwPlus = (((cdkw > as_float(0x2edbe6ffu))) ? (cdkw) : (as_float(0x2edbe6ffu)));
    float ySq = (yC * yC);
    float ySqSafe = (((ySq > as_float(0x1e3ce508u))) ? (ySq) : (as_float(0x1e3ce508u)));
    float kSafe = (((kC > as_float(0x00000000u))) ? (kC) : (as_float(0x00000000u)));
    float sqrtK = sqrt(kSafe);
    float denom1 = (as_float(0x3db851ecu) * (omSafe * yC));
    float denom1Safe = (((denom1 > as_float(0x2edbe6ffu))) ? (denom1) : (as_float(0x2edbe6ffu)));
    float term1 = (sqrtK / denom1Safe);
    float denom2 = (ySqSafe * omSafe);
    float term2 = ((as_float(0x43fa0000u) * nuC) / denom2);
    float maxT12 = (((term1 >= term2)) ? (term1) : (term2));
    float denom3 = (cdkwPlus * ySqSafe);
    float denom3Safe = (((denom3 > as_float(0x2edbe6ffu))) ? (denom3) : (as_float(0x2edbe6ffu)));
    float term3 = ((as_float(0x405b22d1u) * kC) / denom3Safe);
    float minT3 = (((maxT12 <= term3)) ? (maxT12) : (term3));
    float arg1 = (((minT3 <= as_float(0x41200000u))) ? (minT3) : (as_float(0x41200000u)));
    float arg1sq = (arg1 * arg1);
    float arg1p4 = (arg1sq * arg1sq);
    float arg1p4Safe = (((arg1p4 <= as_float(0x42200000u))) ? (arg1p4) : (as_float(0x42200000u)));
    float f1Val = tanh(arg1p4Safe);
    float term2a = ((as_float(0x40000000u) * sqrtK) / denom1Safe);
    float maxF2 = (((term2a >= term2)) ? (term2a) : (term2));
    float arg2 = (((maxF2 <= as_float(0x42c80000u))) ? (maxF2) : (as_float(0x42c80000u)));
    float arg2sq = (arg2 * arg2);
    float arg2sqSafe = (((arg2sq <= as_float(0x42200000u))) ? (arg2sq) : (as_float(0x42200000u)));
    float f23Val = tanh(arg2sqSafe);
    outF1[((int)((int)(get_global_id(0))))] = f1Val;
    outF23[((int)((int)(get_global_id(0))))] = f23Val;
    outCDkOmega[((int)((int)(get_global_id(0))))] = cdkw;
  }
}
