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

kernel void navatala_cfd_n_hatf_all(device const float* gx [[buffer(0)]], device const float* gy [[buffer(1)]], device const float* gz [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const int* owner [[buffer(6)]], device const int* nei [[buffer(7)]], device const float* weights [[buffer(8)]], device const float* theta [[buffer(9)]], device const uint* thetaMask [[buffer(10)]], device const float* paramsF [[buffer(11)]], device const int* counts [[buffer(12)]], device float* contactGrad [[buffer(13)]], device float* nHatf [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    bool isBoundary = (((int)(int(__gid.x))) >= counts[2]);
    float w = weights[((int)(int(__gid.x)))];
    float wm = (as_type<float>(0x3f800000u) - w);
    float gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    float gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    float gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float inv = (as_type<float>(0x3f800000u) / (magg + paramsF[0]));
    float nx = (gfx * inv);
    float ny = (gfy * inv);
    float nz = (gfz * inv);
    contactGrad[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    if (thetaMask[((int)(int(__gid.x)))] != 0u) {
      float sfx = sfX[((int)(int(__gid.x)))];
      float sfy = sfY[((int)(int(__gid.x)))];
      float sfz = sfZ[((int)(int(__gid.x)))];
      float magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > as_type<float>(0x1e3ce508u)) {
        float nfx = (sfx / magSf);
        float nfy = (sfy / magSf);
        float nfz = (sfz / magSf);
        float a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        float a12 = a12raw;
        if (a12 < as_type<float>(0xbf800000u)) {
          a12 = as_type<float>(0xbf800000u);
        } else {
          if (a12 > as_type<float>(0x3f800000u)) {
            a12 = as_type<float>(0x3f800000u);
          }
        }
        float th = theta[((int)(int(__gid.x)))];
        float b1 = cos(th);
        float acosA12 = acos(a12);
        float b2 = cos((acosA12 - th));
        float det = (as_type<float>(0x3f800000u) - (a12 * a12));
        if (det < as_type<float>(0x358637bdu)) {
          det = as_type<float>(0x358637bdu);
        }
        float a = ((b1 - (a12 * b2)) / det);
        float b = ((b2 - (a12 * b1)) / det);
        float nx2 = ((a * nfx) + (b * nx));
        float ny2 = ((a * nfy) + (b * ny));
        float nz2 = ((a * nfz) + (b * nz));
        float magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        float invN = (as_type<float>(0x3f800000u) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        float dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)(int(__gid.x)))] = (dot * magg);
      }
    }
    nHatf[((int)(int(__gid.x)))] = (((nx * sfX[((int)(int(__gid.x)))]) + (ny * sfY[((int)(int(__gid.x)))])) + (nz * sfZ[((int)(int(__gid.x)))]));
  }
}
