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
extern "C" __global__ void navatala_cfd_cf_mesh_volume_optimizer_gradient_f64(const double* pointX, const double* pointY, const double* pointZ, const int* tetConn, const int* tetOffsets, const int* tetCounts, const int* fixedMask, const double* kPerStencil, const int* counts, double* outGradX, double* outGradY, double* outGradZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0) {
      outGradX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __longlong_as_double(0x0000000000000000ull);
      outGradY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __longlong_as_double(0x0000000000000000ull);
      outGradZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __longlong_as_double(0x0000000000000000ull);
    } else {
      double k = kPerStencil[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      double gxSum = __longlong_as_double(0x0000000000000000ull);
      double gySum = __longlong_as_double(0x0000000000000000ull);
      double gzSum = __longlong_as_double(0x0000000000000000ull);
      int tetBeg = tetOffsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      int tetN = tetCounts[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        double ax = pointX[aId];
        double ay = pointY[aId];
        double az = pointZ[aId];
        double bx = pointX[bId];
        double by = pointY[bId];
        double bz = pointZ[bId];
        double cx = pointX[cId];
        double cy = pointY[cId];
        double cz = pointZ[cId];
        double dx = pointX[dId];
        double dy = pointY[dId];
        double dz = pointZ[dId];
        double bax = (bx - ax);
        double bay = (by - ay);
        double baz = (bz - az);
        double cax = (cx - ax);
        double cay = (cy - ay);
        double caz = (cz - az);
        double dax = (dx - ax);
        double day = (dy - ay);
        double daz = (dz - az);
        double dbx = (dx - bx);
        double dby = (dy - by);
        double dbz = (dz - bz);
        double dcx = (dx - cx);
        double dcy = (dy - cy);
        double dcz = (dz - cz);
        double crossX = ((bay * caz) - (baz * cay));
        double crossY = ((baz * cax) - (bax * caz));
        double crossZ = ((bax * cay) - (bay * cax));
        double dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        double vtri = (dotVol / __longlong_as_double(0x4018000000000000ull));
        double lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        double gradVx = (crossX / __longlong_as_double(0x4018000000000000ull));
        double gradVy = (crossY / __longlong_as_double(0x4018000000000000ull));
        double gradVz = (crossZ / __longlong_as_double(0x4018000000000000ull));
        double stab = sqrt(((vtri * vtri) + k));
        double vs = (((vtri < __longlong_as_double(0x0000000000000000ull))) ? (((__longlong_as_double(0x3fe0000000000000ull) * k) / (stab - vtri))) : ((__longlong_as_double(0x3fe0000000000000ull) * (vtri + stab))));
        double gradStabX = ((vtri * gradVx) / stab);
        double gradStabY = ((vtri * gradVy) / stab);
        double gradStabZ = ((vtri * gradVz) / stab);
        double gradVsX = (((vtri < __longlong_as_double(0x0000000000000000ull))) ? (((__longlong_as_double(0xbfe0000000000000ull) * (k * (gradStabX - gradVx))) / ((stab - vtri) * (stab - vtri)))) : ((__longlong_as_double(0x3fe0000000000000ull) * (gradVx + gradStabX))));
        double gradVsY = (((vtri < __longlong_as_double(0x0000000000000000ull))) ? (((__longlong_as_double(0xbfe0000000000000ull) * (k * (gradStabY - gradVy))) / ((stab - vtri) * (stab - vtri)))) : ((__longlong_as_double(0x3fe0000000000000ull) * (gradVy + gradStabY))));
        double gradVsZ = (((vtri < __longlong_as_double(0x0000000000000000ull))) ? (((__longlong_as_double(0xbfe0000000000000ull) * (k * (gradStabZ - gradVz))) / ((stab - vtri) * (stab - vtri)))) : ((__longlong_as_double(0x3fe0000000000000ull) * (gradVz + gradStabZ))));
        double gradLsqX = (__longlong_as_double(0x4000000000000000ull) * ((__longlong_as_double(0x4008000000000000ull) * dx) - ((ax + bx) + cx)));
        double gradLsqY = (__longlong_as_double(0x4000000000000000ull) * ((__longlong_as_double(0x4008000000000000ull) * dy) - ((ay + by) + cy)));
        double gradLsqZ = (__longlong_as_double(0x4000000000000000ull) * ((__longlong_as_double(0x4008000000000000ull) * dz) - ((az + bz) + cz)));
        double c0 = (__longlong_as_double(0x3fe5555555555555ull) * pow(__longlong_as_double(0x3fe0000000000000ull), __longlong_as_double(0x3fe5555555555555ull)));
        double vs13 = pow((__longlong_as_double(0x4000000000000000ull) * vs), __longlong_as_double(0x3fd5555555555555ull));
        double vstab = pow(vs, __longlong_as_double(0x3fe5555555555555ull));
        double sqrVstab = (vstab * vstab);
        double gradVstabX = ((c0 * (__longlong_as_double(0x4000000000000000ull) * gradVsX)) / vs13);
        double gradVstabY = ((c0 * (__longlong_as_double(0x4000000000000000ull) * gradVsY)) / vs13);
        double gradVstabZ = ((c0 * (__longlong_as_double(0x4000000000000000ull) * gradVsZ)) / vs13);
        double termLX = (gradLsqX / vstab);
        double termLY = (gradLsqY / vstab);
        double termLZ = (gradLsqZ / vstab);
        double termVX = ((lsqr * gradVstabX) / sqrVstab);
        double termVY = ((lsqr * gradVstabY) / sqrVstab);
        double termVZ = ((lsqr * gradVstabZ) / sqrVstab);
        double gx = (termLX - termVX);
        double gy = (termLY - termVY);
        double gz = (termLZ - termVZ);
        gxSum = (gxSum + gx);
        gySum = (gySum + gy);
        gzSum = (gzSum + gz);
      }
      outGradX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = gxSum;
      outGradY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = gySum;
      outGradZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = gzSum;
    }
  }
}
