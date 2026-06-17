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

__kernel void navatala_cfd_n_hatf_all(__global const float* gx, __global const float* gy, __global const float* gz, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* owner, __global const int* nei, __global const float* weights, __global const float* theta, __global const uint* thetaMask, __global const float* paramsF, __global const int* counts, __global float* contactGrad, __global float* nHatf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    bool isBoundary = (((int)((int)(get_global_id(0)))) >= counts[2]);
    float w = weights[((int)((int)(get_global_id(0))))];
    float wm = (as_float(0x3f800000u) - w);
    float gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    float gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    float gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float inv = (as_float(0x3f800000u) / (magg + paramsF[0]));
    float nx = (gfx * inv);
    float ny = (gfy * inv);
    float nz = (gfz * inv);
    contactGrad[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    if ((thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u))) {
      float sfx = sfX[((int)((int)(get_global_id(0))))];
      float sfy = sfY[((int)((int)(get_global_id(0))))];
      float sfz = sfZ[((int)((int)(get_global_id(0))))];
      float magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if ((magSf > as_float(0x1e3ce508u))) {
        float nfx = (sfx / magSf);
        float nfy = (sfy / magSf);
        float nfz = (sfz / magSf);
        float a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        float a12 = a12raw;
        if ((a12 < as_float(0xbf800000u))) {
          a12 = as_float(0xbf800000u);
        } else {
          if ((a12 > as_float(0x3f800000u))) {
            a12 = as_float(0x3f800000u);
          }
        }
        float th = theta[((int)((int)(get_global_id(0))))];
        float b1 = cos(th);
        float acosA12 = acos(a12);
        float b2 = cos((acosA12 - th));
        float det = (as_float(0x3f800000u) - (a12 * a12));
        if ((det < as_float(0x358637bdu))) {
          det = as_float(0x358637bdu);
        }
        float a = ((b1 - (a12 * b2)) / det);
        float b = ((b2 - (a12 * b1)) / det);
        float nx2 = ((a * nfx) + (b * nx));
        float ny2 = ((a * nfy) + (b * ny));
        float nz2 = ((a * nfz) + (b * nz));
        float magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        float invN = (as_float(0x3f800000u) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        float dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(get_global_id(0))))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(get_global_id(0))))] = (((nx * sfX[((int)((int)(get_global_id(0))))]) + (ny * sfY[((int)((int)(get_global_id(0))))])) + (nz * sfZ[((int)((int)(get_global_id(0))))]));
  }
}
