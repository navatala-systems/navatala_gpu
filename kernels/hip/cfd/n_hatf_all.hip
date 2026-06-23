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
extern "C" __global__ void navatala_cfd_n_hatf_all(const float* gx, const float* gy, const float* gz, const float* sfX, const float* sfY, const float* sfZ, const int* owner, const int* nei, const float* weights, const float* theta, const unsigned int* thetaMask, const float* paramsF, const int* counts, float* contactGrad, float* nHatf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    bool isBoundary = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]);
    float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float wm = (__uint_as_float(0x3f800000u) - w);
    float gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    float gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    float gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float inv = (__uint_as_float(0x3f800000u) / (magg + paramsF[0]));
    float nx = (gfx * inv);
    float ny = (gfy * inv);
    float nz = (gfz * inv);
    contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    if ((thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u)) {
      float sfx = sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float sfy = sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float sfz = sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if ((magSf > __uint_as_float(0x1e3ce508u))) {
        float nfx = (sfx / magSf);
        float nfy = (sfy / magSf);
        float nfz = (sfz / magSf);
        float a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        float a12 = a12raw;
        if ((a12 < __uint_as_float(0xbf800000u))) {
          a12 = __uint_as_float(0xbf800000u);
        } else {
          if ((a12 > __uint_as_float(0x3f800000u))) {
            a12 = __uint_as_float(0x3f800000u);
          }
        }
        float th = theta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float b1 = cos(th);
        float acosA12 = acos(a12);
        float b2 = cos((acosA12 - th));
        float det = (__uint_as_float(0x3f800000u) - (a12 * a12));
        if ((det < __uint_as_float(0x358637bdu))) {
          det = __uint_as_float(0x358637bdu);
        }
        float a = ((b1 - (a12 * b2)) / det);
        float b = ((b2 - (a12 * b1)) / det);
        float nx2 = ((a * nfx) + (b * nx));
        float ny2 = ((a * nfy) + (b * ny));
        float nz2 = ((a * nfz) + (b * nz));
        float magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        float invN = (__uint_as_float(0x3f800000u) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        float dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((nx * sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) + (ny * sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) + (nz * sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}
