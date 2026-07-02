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
__kernel void navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64(__global const double* pointX, __global const double* pointY, __global const double* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const double* params, __global const int* counts, __global double* outK) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outK[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    } else {
      double small = params[0];
      double vMin = as_double(0x7e37e43c8800759cul);
      double lSqrMax = as_double(0x0000000000000000ul);
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
        vMin = (((vtri < vMin)) ? (vtri) : (vMin));
        lSqrMax = (((lsqr > lSqrMax)) ? (lsqr) : (lSqrMax));
      }
      double threshold = (small * lSqrMax);
      double kVal = (((vMin < threshold)) ? (threshold) : (as_double(0x0000000000000000ul)));
      outK[((int)((int)(get_global_id(0))))] = kVal;
    }
  }
}
