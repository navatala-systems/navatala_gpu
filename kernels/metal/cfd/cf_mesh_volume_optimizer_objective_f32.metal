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

kernel void navatala_cfd_cf_mesh_volume_optimizer_objective_f32(device const float* pointX [[buffer(0)]], device const float* pointY [[buffer(1)]], device const float* pointZ [[buffer(2)]], device const int* tetConn [[buffer(3)]], device const int* tetOffsets [[buffer(4)]], device const int* tetCounts [[buffer(5)]], device const int* fixedMask [[buffer(6)]], device const float* kPerStencil [[buffer(7)]], device const int* counts [[buffer(8)]], device float* outObjective [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)(int(__gid.x)))] != 0) {
      outObjective[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    } else {
      float k = kPerStencil[((int)(int(__gid.x)))];
      float sum = as_type<float>(0x00000000u);
      int tetBeg = tetOffsets[((int)(int(__gid.x)))];
      int tetN = tetCounts[((int)(int(__gid.x)))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        float ax = pointX[aId];
        float ay = pointY[aId];
        float az = pointZ[aId];
        float bx = pointX[bId];
        float by = pointY[bId];
        float bz = pointZ[bId];
        float cx = pointX[cId];
        float cy = pointY[cId];
        float cz = pointZ[cId];
        float dx = pointX[dId];
        float dy = pointY[dId];
        float dz = pointZ[dId];
        float bax = (bx - ax);
        float bay = (by - ay);
        float baz = (bz - az);
        float cax = (cx - ax);
        float cay = (cy - ay);
        float caz = (cz - az);
        float dax = (dx - ax);
        float day = (dy - ay);
        float daz = (dz - az);
        float dbx = (dx - bx);
        float dby = (dy - by);
        float dbz = (dz - bz);
        float dcx = (dx - cx);
        float dcy = (dy - cy);
        float dcz = (dz - cz);
        float crossX = ((bay * caz) - (baz * cay));
        float crossY = ((baz * cax) - (bax * caz));
        float crossZ = ((bax * cay) - (bay * cax));
        float dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        float vtri = (dotVol / as_type<float>(0x40c00000u));
        float lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        float sqrtArg = ((vtri * vtri) + k);
        float stab = sqrt(sqrtArg);
        float vstab = (((vtri < as_type<float>(0x00000000u))) ? (((as_type<float>(0x3f000000u) * k) / (stab - vtri))) : ((as_type<float>(0x3f000000u) * (vtri + stab))));
        float denom = pow(vstab, as_type<float>(0x3f2aaaabu));
        float contrib = (lsqr / denom);
        sum = (sum + contrib);
      }
      outObjective[((int)(int(__gid.x)))] = sum;
    }
  }
}
