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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_n_hatf_all_f64(const double* gx, const double* gy, const double* gz, const double* sfX, const double* sfY, const double* sfZ, const int* owner, const int* nei, const double* weights, const float* theta, const unsigned int* thetaMask, const double* paramsF, const int* counts, double* contactGrad, double* nHatf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    bool isBoundary = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]);
    double w = ((double)(weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    double wm = (__longlong_as_double(0x3ff0000000000000ull) - w);
    double gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    double gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    double gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    double magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    double inv = (__longlong_as_double(0x3ff0000000000000ull) / (magg + paramsF[0]));
    double nx = (gfx * inv);
    double ny = (gfy * inv);
    double nz = (gfz * inv);
    contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __longlong_as_double(0x0000000000000000ull);
    if (thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u) {
      double sfx = ((double)(sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      double sfy = ((double)(sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      double sfz = ((double)(sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      double magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > __longlong_as_double(0x3bc79ca10c924223ull)) {
        double nfx = (sfx / magSf);
        double nfy = (sfy / magSf);
        double nfz = (sfz / magSf);
        double a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        double a12 = a12raw;
        if (a12 < __longlong_as_double(0xbff0000000000000ull)) {
          a12 = __longlong_as_double(0xbff0000000000000ull);
        } else {
          if (a12 > __longlong_as_double(0x3ff0000000000000ull)) {
            a12 = __longlong_as_double(0x3ff0000000000000ull);
          }
        }
        double th = ((double)(theta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        double b1 = cos(th);
        double acosA12 = acos(a12);
        double b2 = cos((acosA12 - th));
        double det = (__longlong_as_double(0x3ff0000000000000ull) - (a12 * a12));
        if (det < __longlong_as_double(0x3eb0c6f7a0b5ed8dull)) {
          det = __longlong_as_double(0x3eb0c6f7a0b5ed8dull);
        }
        double a = ((b1 - (a12 * b2)) / det);
        double b = ((b2 - (a12 * b1)) / det);
        double nx2 = ((a * nfx) + (b * nx));
        double ny2 = ((a * nfy) + (b * ny));
        double nz2 = ((a * nfz) + (b * nz));
        double magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        double invN = (__longlong_as_double(0x3ff0000000000000ull) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        double dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((nx * ((double)(sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]))) + (ny * ((double)(sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])))) + (nz * ((double)(sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]))));
  }
}
