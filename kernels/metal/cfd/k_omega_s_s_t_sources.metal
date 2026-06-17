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

kernel void navatala_cfd_k_omega_s_s_t_sources(device const float* k [[buffer(0)]], device const float* omega [[buffer(1)]], device const float* nut [[buffer(2)]], device const float* nu [[buffer(3)]], device const float* alpha [[buffer(4)]], device const float* rho [[buffer(5)]], device const float* F1 [[buffer(6)]], device const float* F23 [[buffer(7)]], device const float* CDkOmega [[buffer(8)]], device const float* S2 [[buffer(9)]], device const float* divU [[buffer(10)]], device const int* counts [[buffer(11)]], device float* outOmegaSp [[buffer(12)]], device float* outOmegaSuSp1 [[buffer(13)]], device float* outOmegaSuSp2 [[buffer(14)]], device float* outOmegaExpl [[buffer(15)]], device float* outKSp [[buffer(16)]], device float* outKSuSp [[buffer(17)]], device float* outKExpl [[buffer(18)]], device float* outDomegaEff [[buffer(19)]], device float* outDkEff [[buffer(20)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float kC = k[((int)(int(__gid.x)))];
    float omC = omega[((int)(int(__gid.x)))];
    float nutC = nut[((int)(int(__gid.x)))];
    float nuC = nu[((int)(int(__gid.x)))];
    float alphaC = alpha[((int)(int(__gid.x)))];
    float rhoC = rho[((int)(int(__gid.x)))];
    float f1C = F1[((int)(int(__gid.x)))];
    float f23C = F23[((int)(int(__gid.x)))];
    float cdkwC = CDkOmega[((int)(int(__gid.x)))];
    float s2C = S2[((int)(int(__gid.x)))];
    float divUC = divU[((int)(int(__gid.x)))];
    float arC = (alphaC * rhoC);
    float gammaC = ((f1C * as_type<float>(0x3decbfb1u)) + as_type<float>(0x3ee147aeu));
    float betaC = (as_type<float>(0x3da9930cu) - (f1C * as_type<float>(0x3bff9724u)));
    float alphaKC = (as_type<float>(0x3f800000u) - (f1C * as_type<float>(0x3e19999au)));
    float alphaOmC = (as_type<float>(0x3f5b22d1u) - (f1C * as_type<float>(0x3eb645a2u)));
    float s2Safe = (((s2C > as_type<float>(0x00000000u))) ? (s2C) : (as_type<float>(0x00000000u)));
    float sqrtS2 = sqrt(s2Safe);
    float a1om = (as_type<float>(0x3e9eb852u) * omC);
    float f23s = (f23C * sqrtS2);
    float maxTerm = (((a1om >= f23s)) ? (a1om) : (f23s));
    float limRhs = (as_type<float>(0x4039ce07u) * (omC * maxTerm));
    float GbyNu = (((s2C <= limRhs)) ? (s2C) : (limRhs));
    float nutS2 = (nutC * s2C);
    float c1bsKOm = (as_type<float>(0x3f666666u) * (kC * omC));
    float pk = (((nutS2 <= c1bsKOm)) ? (nutS2) : (c1bsKOm));
    float omSp = (as_type<float>(0x00000000u) - (arC * (betaC * omC)));
    float omSuSp1 = (as_type<float>(0x00000000u) - (as_type<float>(0x3f2aacdau) * (arC * (gammaC * divUC))));
    float omSafe = (((omC > as_type<float>(0x26901d7du))) ? (omC) : (as_type<float>(0x26901d7du)));
    float f1m1 = (f1C - as_type<float>(0x3f800000u));
    float omSuSp2 = (as_type<float>(0x00000000u) - ((arC * (f1m1 * cdkwC)) / omSafe));
    float omExpl = (arC * (gammaC * GbyNu));
    float kSp = (as_type<float>(0x00000000u) - (arC * (as_type<float>(0x3db851ecu) * omC)));
    float kSuSp = (as_type<float>(0x00000000u) - (as_type<float>(0x3f2aacdau) * (arC * divUC)));
    float kExpl = (arC * pk);
    float domEff = (arC * ((alphaOmC * nutC) + nuC));
    float dkEff = (arC * ((alphaKC * nutC) + nuC));
    outOmegaSp[((int)(int(__gid.x)))] = omSp;
    outOmegaSuSp1[((int)(int(__gid.x)))] = omSuSp1;
    outOmegaSuSp2[((int)(int(__gid.x)))] = omSuSp2;
    outOmegaExpl[((int)(int(__gid.x)))] = omExpl;
    outKSp[((int)(int(__gid.x)))] = kSp;
    outKSuSp[((int)(int(__gid.x)))] = kSuSp;
    outKExpl[((int)(int(__gid.x)))] = kExpl;
    outDomegaEff[((int)(int(__gid.x)))] = domEff;
    outDkEff[((int)(int(__gid.x)))] = dkEff;
  }
}
