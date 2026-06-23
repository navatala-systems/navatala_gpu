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

kernel void navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp(device const float* rhoCell [[buffer(0)]], device const float* muCell [[buffer(1)]], device const float* phiFaces [[buffer(2)]], device const float* rhoBcVal [[buffer(3)]], device const uint* rhoBcMask [[buffer(4)]], device const float* muBcVal [[buffer(5)]], device const uint* muBcMask [[buffer(6)]], device const int* ownerAll [[buffer(7)]], device const int* neighbourInt [[buffer(8)]], device const float* weightsAll [[buffer(9)]], device const float* magSfAll [[buffer(10)]], device const float* deltaAll [[buffer(11)]], device const uint* offsets [[buffer(12)]], device const uint* faceIdx [[buffer(13)]], device const int* sign [[buffer(14)]], device const float* volCells [[buffer(15)]], device const float* sp [[buffer(16)]], device const int* counts3 [[buffer(17)]], device const float* paramsF [[buffer(18)]], device float* outRAU [[buffer(19)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts3[0]) {
    return;
  } else {
    float invDt = (as_type<float>(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)(int(__gid.x)))];
    float A = (rhoCell[((int)(int(__gid.x)))] * invDt);
    float conv = as_type<float>(0x00000000u);
    float diff = as_type<float>(0x00000000u);
    uint begU = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      if (f >= counts3[1]) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = as_type<float>(0x00000000u);
        float muF = as_type<float>(0x00000000u);
        if (f < counts3[2]) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (as_type<float>(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if (rhoBcMask[f] != 0u) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if (muBcMask[f] != 0u) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if (s < 0) {
          outFlux = (as_type<float>(0x00000000u) - rhoPhi);
        }
        if (outFlux > as_type<float>(0x00000000u)) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    A = (A + sp[((int)(int(__gid.x)))]);
    if (A != as_type<float>(0x00000000u)) {
      outRAU[((int)(int(__gid.x)))] = (as_type<float>(0x3f800000u) / A);
    } else {
      outRAU[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}
