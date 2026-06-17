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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_k_omega_s_s_t_blending(device const float* k [[buffer(0)]], device const float* omega [[buffer(1)]], device const float* nut [[buffer(2)]], device const float* nu [[buffer(3)]], device const float* y [[buffer(4)]], device const float* gradKx [[buffer(5)]], device const float* gradKy [[buffer(6)]], device const float* gradKz [[buffer(7)]], device const float* gradOmegax [[buffer(8)]], device const float* gradOmegay [[buffer(9)]], device const float* gradOmegaz [[buffer(10)]], device const int* counts [[buffer(11)]], device float* outF1 [[buffer(12)]], device float* outF23 [[buffer(13)]], device float* outCDkOmega [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float kC = k[((int)(int(__gid.x)))];
    float omC = omega[((int)(int(__gid.x)))];
    float nutC = nut[((int)(int(__gid.x)))];
    float nuC = nu[((int)(int(__gid.x)))];
    float yC = y[((int)(int(__gid.x)))];
    float gKx = gradKx[((int)(int(__gid.x)))];
    float gKy = gradKy[((int)(int(__gid.x)))];
    float gKz = gradKz[((int)(int(__gid.x)))];
    float gOx = gradOmegax[((int)(int(__gid.x)))];
    float gOy = gradOmegay[((int)(int(__gid.x)))];
    float gOz = gradOmegaz[((int)(int(__gid.x)))];
    float dotGrad = (((gKx * gOx) + (gKy * gOy)) + (gKz * gOz));
    float omSafe = (((omC > as_type<float>(0x2edbe6ffu))) ? (omC) : (as_type<float>(0x2edbe6ffu)));
    float cdkw = ((as_type<float>(0x3fdb22d1u) * dotGrad) / omSafe);
    float cdkwPlus = (((cdkw > as_type<float>(0x2edbe6ffu))) ? (cdkw) : (as_type<float>(0x2edbe6ffu)));
    float ySq = (yC * yC);
    float ySqSafe = (((ySq > as_type<float>(0x1e3ce508u))) ? (ySq) : (as_type<float>(0x1e3ce508u)));
    float kSafe = (((kC > as_type<float>(0x00000000u))) ? (kC) : (as_type<float>(0x00000000u)));
    float sqrtK = sqrt(kSafe);
    float denom1 = (as_type<float>(0x3db851ecu) * (omSafe * yC));
    float denom1Safe = (((denom1 > as_type<float>(0x2edbe6ffu))) ? (denom1) : (as_type<float>(0x2edbe6ffu)));
    float term1 = (sqrtK / denom1Safe);
    float denom2 = (ySqSafe * omSafe);
    float term2 = ((as_type<float>(0x43fa0000u) * nuC) / denom2);
    float maxT12 = (((term1 >= term2)) ? (term1) : (term2));
    float denom3 = (cdkwPlus * ySqSafe);
    float denom3Safe = (((denom3 > as_type<float>(0x2edbe6ffu))) ? (denom3) : (as_type<float>(0x2edbe6ffu)));
    float term3 = ((as_type<float>(0x405b22d1u) * kC) / denom3Safe);
    float minT3 = (((maxT12 <= term3)) ? (maxT12) : (term3));
    float arg1 = (((minT3 <= as_type<float>(0x41200000u))) ? (minT3) : (as_type<float>(0x41200000u)));
    float arg1sq = (arg1 * arg1);
    float arg1p4 = (arg1sq * arg1sq);
    float arg1p4Safe = (((arg1p4 <= as_type<float>(0x42200000u))) ? (arg1p4) : (as_type<float>(0x42200000u)));
    float f1Val = tanh(arg1p4Safe);
    float term2a = ((as_type<float>(0x40000000u) * sqrtK) / denom1Safe);
    float maxF2 = (((term2a >= term2)) ? (term2a) : (term2));
    float arg2 = (((maxF2 <= as_type<float>(0x42c80000u))) ? (maxF2) : (as_type<float>(0x42c80000u)));
    float arg2sq = (arg2 * arg2);
    float arg2sqSafe = (((arg2sq <= as_type<float>(0x42200000u))) ? (arg2sq) : (as_type<float>(0x42200000u)));
    float f23Val = tanh(arg2sqSafe);
    outF1[((int)(int(__gid.x)))] = f1Val;
    outF23[((int)(int(__gid.x)))] = f23Val;
    outCDkOmega[((int)(int(__gid.x)))] = cdkw;
  }
}
