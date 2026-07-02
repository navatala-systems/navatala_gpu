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
extern "C" __global__ void navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32(const float* pointX, const float* pointY, const float* pointZ, const int* tetConn, const int* tetOffsets, const int* tetCounts, const int* fixedMask, const float* params, const int* counts, float* outK) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0) {
      outK[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    } else {
      float small = params[0];
      float vMin = __uint_as_float(0x7149f2cau);
      float lSqrMax = __uint_as_float(0x00000000u);
      int tetBeg = tetOffsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      int tetN = tetCounts[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
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
        float vtri = (dotVol / __uint_as_float(0x40c00000u));
        float lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        vMin = (((vtri < vMin)) ? (vtri) : (vMin));
        lSqrMax = (((lsqr > lSqrMax)) ? (lsqr) : (lSqrMax));
      }
      float threshold = (small * lSqrMax);
      float kVal = (((vMin < threshold)) ? (threshold) : (__uint_as_float(0x00000000u)));
      outK[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kVal;
    }
  }
}
