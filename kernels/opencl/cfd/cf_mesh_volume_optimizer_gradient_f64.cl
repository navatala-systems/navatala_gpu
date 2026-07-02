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
__kernel void navatala_cfd_cf_mesh_volume_optimizer_gradient_f64(__global const double* pointX, __global const double* pointY, __global const double* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const double* kPerStencil, __global const int* counts, __global double* outGradX, __global double* outGradY, __global double* outGradZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outGradX[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
      outGradY[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
      outGradZ[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    } else {
      double k = kPerStencil[((int)((int)(get_global_id(0))))];
      double gxSum = as_double(0x0000000000000000ul);
      double gySum = as_double(0x0000000000000000ul);
      double gzSum = as_double(0x0000000000000000ul);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
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
        double vtri = (dotVol / as_double(0x4018000000000000ul));
        double lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        double gradVx = (crossX / as_double(0x4018000000000000ul));
        double gradVy = (crossY / as_double(0x4018000000000000ul));
        double gradVz = (crossZ / as_double(0x4018000000000000ul));
        double stab = sqrt(((vtri * vtri) + k));
        double vs = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0x3fe0000000000000ul) * k) / (stab - vtri))) : ((as_double(0x3fe0000000000000ul) * (vtri + stab))));
        double gradStabX = ((vtri * gradVx) / stab);
        double gradStabY = ((vtri * gradVy) / stab);
        double gradStabZ = ((vtri * gradVz) / stab);
        double gradVsX = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0xbfe0000000000000ul) * (k * (gradStabX - gradVx))) / ((stab - vtri) * (stab - vtri)))) : ((as_double(0x3fe0000000000000ul) * (gradVx + gradStabX))));
        double gradVsY = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0xbfe0000000000000ul) * (k * (gradStabY - gradVy))) / ((stab - vtri) * (stab - vtri)))) : ((as_double(0x3fe0000000000000ul) * (gradVy + gradStabY))));
        double gradVsZ = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0xbfe0000000000000ul) * (k * (gradStabZ - gradVz))) / ((stab - vtri) * (stab - vtri)))) : ((as_double(0x3fe0000000000000ul) * (gradVz + gradStabZ))));
        double gradLsqX = (as_double(0x4000000000000000ul) * ((as_double(0x4008000000000000ul) * dx) - ((ax + bx) + cx)));
        double gradLsqY = (as_double(0x4000000000000000ul) * ((as_double(0x4008000000000000ul) * dy) - ((ay + by) + cy)));
        double gradLsqZ = (as_double(0x4000000000000000ul) * ((as_double(0x4008000000000000ul) * dz) - ((az + bz) + cz)));
        double c0 = (as_double(0x3fe5555555555555ul) * pow(as_double(0x3fe0000000000000ul), as_double(0x3fe5555555555555ul)));
        double vs13 = pow((as_double(0x4000000000000000ul) * vs), as_double(0x3fd5555555555555ul));
        double vstab = pow(vs, as_double(0x3fe5555555555555ul));
        double sqrVstab = (vstab * vstab);
        double gradVstabX = ((c0 * (as_double(0x4000000000000000ul) * gradVsX)) / vs13);
        double gradVstabY = ((c0 * (as_double(0x4000000000000000ul) * gradVsY)) / vs13);
        double gradVstabZ = ((c0 * (as_double(0x4000000000000000ul) * gradVsZ)) / vs13);
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
      outGradX[((int)((int)(get_global_id(0))))] = gxSum;
      outGradY[((int)((int)(get_global_id(0))))] = gySum;
      outGradZ[((int)((int)(get_global_id(0))))] = gzSum;
    }
  }
}
