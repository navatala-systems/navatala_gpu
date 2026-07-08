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

kernel void navatala_cfd_vof_alpha_phi_van_leer_int(device const float* alpha [[buffer(0)]], device const float* phi [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device const float* gradX [[buffer(4)]], device const float* gradY [[buffer(5)]], device const float* gradZ [[buffer(6)]], device const float* cellCx [[buffer(7)]], device const float* cellCy [[buffer(8)]], device const float* cellCz [[buffer(9)]], device const float* weights [[buffer(10)]], device const float* phir [[buffer(11)]], device const float* alphaF [[buffer(12)]], device const int* faceParams [[buffer(13)]], device float* outAlphaPhi [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)(int(__gid.x)))];
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float aO = alpha[o];
    float aN = alpha[n];
    float gradf = (aN - aO);
    float gxU = gradX[n];
    float gyU = gradY[n];
    float gzU = gradZ[n];
    if (ph > as_type<float>(0x00000000u)) {
      gxU = gradX[o];
      gyU = gradY[o];
      gzU = gradZ[o];
    }
    float dx = (cellCx[n] - cellCx[o]);
    float dy = (cellCy[n] - cellCy[o]);
    float dz = (cellCz[n] - cellCz[o]);
    float gradcf = (((dx * gxU) + (dy * gyU)) + (dz * gzU));
    float absGradf = abs(gradf);
    float absGradcf = abs(gradcf);
    float gradfSign = as_type<float>(0x3f800000u);
    if (gradf < as_type<float>(0x00000000u)) {
      gradfSign = as_type<float>(0xbf800000u);
    }
    float gradcfSign = as_type<float>(0x3f800000u);
    if (gradcf < as_type<float>(0x00000000u)) {
      gradcfSign = as_type<float>(0xbf800000u);
    }
    float r = as_type<float>(0x00000000u);
    if (absGradf > as_type<float>(0x0da24260u)) {
      if (absGradcf >= (as_type<float>(0x447a0000u) * absGradf)) {
        r = ((as_type<float>(0x44fa0000u) * (gradcfSign * gradfSign)) - as_type<float>(0x3f800000u));
      } else {
        r = ((as_type<float>(0x40000000u) * (gradcf / gradf)) - as_type<float>(0x3f800000u));
      }
    }
    float absR = abs(r);
    float psi = ((r + absR) / (as_type<float>(0x3f800000u) + absR));
    if (((ph > as_type<float>(0x00000000u)) && ((aO < as_type<float>(0x00000000u)) || (aN > as_type<float>(0x3f800000u)))) || ((ph < as_type<float>(0x00000000u)) && ((aN < as_type<float>(0x00000000u)) || (aO > as_type<float>(0x3f800000u))))) {
      psi = as_type<float>(0x00000000u);
    }
    float pos0 = as_type<float>(0x00000000u);
    if (ph >= as_type<float>(0x00000000u)) {
      pos0 = as_type<float>(0x3f800000u);
    }
    float cdw = weights[((int)(int(__gid.x)))];
    float w = ((psi * cdw) + ((as_type<float>(0x3f800000u) - psi) * pos0));
    float oneMinusW = (as_type<float>(0x3f800000u) - w);
    float a = ((w * aO) + (oneMinusW * aN));
    float ac = alphaF[((int)(int(__gid.x)))];
    if (ac < as_type<float>(0x00000000u)) {
      ac = as_type<float>(0x00000000u);
    }
    if (ac > as_type<float>(0x3f800000u)) {
      ac = as_type<float>(0x3f800000u);
    }
    float oneMinusAc = (as_type<float>(0x3f800000u) - ac);
    float comp = (ac * oneMinusAc);
    outAlphaPhi[((int)(int(__gid.x)))] = ((ph * a) + (phir[((int)(int(__gid.x)))] * comp));
  }
}
