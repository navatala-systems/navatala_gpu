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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_n_hatf_all_f64(__global const double* gx, __global const double* gy, __global const double* gz, __global const double* sfX, __global const double* sfY, __global const double* sfZ, __global const int* owner, __global const int* nei, __global const double* weights, __global const float* theta, __global const uint* thetaMask, __global const double* paramsF, __global const int* counts, __global double* contactGrad, __global double* nHatf) {
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
    double w = ((double)(weights[((int)((int)(get_global_id(0))))]));
    double wm = (as_double(0x3ff0000000000000ul) - w);
    double gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    double gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    double gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    double magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    double inv = (as_double(0x3ff0000000000000ul) / (magg + paramsF[0]));
    double nx = (gfx * inv);
    double ny = (gfy * inv);
    double nz = (gfz * inv);
    contactGrad[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    if ((thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u))) {
      double sfx = ((double)(sfX[((int)((int)(get_global_id(0))))]));
      double sfy = ((double)(sfY[((int)((int)(get_global_id(0))))]));
      double sfz = ((double)(sfZ[((int)((int)(get_global_id(0))))]));
      double magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if ((magSf > as_double(0x3bc79ca10c924223ul))) {
        double nfx = (sfx / magSf);
        double nfy = (sfy / magSf);
        double nfz = (sfz / magSf);
        double a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        double a12 = a12raw;
        if ((a12 < as_double(0xbff0000000000000ul))) {
          a12 = as_double(0xbff0000000000000ul);
        } else {
          if ((a12 > as_double(0x3ff0000000000000ul))) {
            a12 = as_double(0x3ff0000000000000ul);
          }
        }
        double th = ((double)(theta[((int)((int)(get_global_id(0))))]));
        double b1 = cos(th);
        double acosA12 = acos(a12);
        double b2 = cos((acosA12 - th));
        double det = (as_double(0x3ff0000000000000ul) - (a12 * a12));
        if ((det < as_double(0x3eb0c6f7a0b5ed8dul))) {
          det = as_double(0x3eb0c6f7a0b5ed8dul);
        }
        double a = ((b1 - (a12 * b2)) / det);
        double b = ((b2 - (a12 * b1)) / det);
        double nx2 = ((a * nfx) + (b * nx));
        double ny2 = ((a * nfy) + (b * ny));
        double nz2 = ((a * nfz) + (b * nz));
        double magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        double invN = (as_double(0x3ff0000000000000ul) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        double dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(get_global_id(0))))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(get_global_id(0))))] = (((nx * ((double)(sfX[((int)((int)(get_global_id(0))))]))) + (ny * ((double)(sfY[((int)((int)(get_global_id(0))))])))) + (nz * ((double)(sfZ[((int)((int)(get_global_id(0))))]))));
  }
}
