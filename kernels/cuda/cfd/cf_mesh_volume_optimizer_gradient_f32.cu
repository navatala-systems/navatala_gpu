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
extern "C" __global__ void navatala_cfd_cf_mesh_volume_optimizer_gradient_f32(const float* pointX, const float* pointY, const float* pointZ, const int* tetConn, const int* tetOffsets, const int* tetCounts, const int* fixedMask, const float* kPerStencil, const int* counts, float* outGradX, float* outGradY, float* outGradZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0) {
      outGradX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      outGradY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      outGradZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    } else {
      float k = kPerStencil[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float gxSum = __uint_as_float(0x00000000u);
      float gySum = __uint_as_float(0x00000000u);
      float gzSum = __uint_as_float(0x00000000u);
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
        float gradVx = (crossX / __uint_as_float(0x40c00000u));
        float gradVy = (crossY / __uint_as_float(0x40c00000u));
        float gradVz = (crossZ / __uint_as_float(0x40c00000u));
        float stab = sqrt(((vtri * vtri) + k));
        float vs = (((vtri < __uint_as_float(0x00000000u))) ? (((__uint_as_float(0x3f000000u) * k) / (stab - vtri))) : ((__uint_as_float(0x3f000000u) * (vtri + stab))));
        float gradStabX = ((vtri * gradVx) / stab);
        float gradStabY = ((vtri * gradVy) / stab);
        float gradStabZ = ((vtri * gradVz) / stab);
        float gradVsX = (((vtri < __uint_as_float(0x00000000u))) ? (((__uint_as_float(0xbf000000u) * (k * (gradStabX - gradVx))) / ((stab - vtri) * (stab - vtri)))) : ((__uint_as_float(0x3f000000u) * (gradVx + gradStabX))));
        float gradVsY = (((vtri < __uint_as_float(0x00000000u))) ? (((__uint_as_float(0xbf000000u) * (k * (gradStabY - gradVy))) / ((stab - vtri) * (stab - vtri)))) : ((__uint_as_float(0x3f000000u) * (gradVy + gradStabY))));
        float gradVsZ = (((vtri < __uint_as_float(0x00000000u))) ? (((__uint_as_float(0xbf000000u) * (k * (gradStabZ - gradVz))) / ((stab - vtri) * (stab - vtri)))) : ((__uint_as_float(0x3f000000u) * (gradVz + gradStabZ))));
        float gradLsqX = (__uint_as_float(0x40000000u) * ((__uint_as_float(0x40400000u) * dx) - ((ax + bx) + cx)));
        float gradLsqY = (__uint_as_float(0x40000000u) * ((__uint_as_float(0x40400000u) * dy) - ((ay + by) + cy)));
        float gradLsqZ = (__uint_as_float(0x40000000u) * ((__uint_as_float(0x40400000u) * dz) - ((az + bz) + cz)));
        float c0 = (__uint_as_float(0x3f2aaaabu) * powf(__uint_as_float(0x3f000000u), __uint_as_float(0x3f2aaaabu)));
        float vs13 = powf((__uint_as_float(0x40000000u) * vs), __uint_as_float(0x3eaaaaabu));
        float vstab = powf(vs, __uint_as_float(0x3f2aaaabu));
        float sqrVstab = (vstab * vstab);
        float gradVstabX = ((c0 * (__uint_as_float(0x40000000u) * gradVsX)) / vs13);
        float gradVstabY = ((c0 * (__uint_as_float(0x40000000u) * gradVsY)) / vs13);
        float gradVstabZ = ((c0 * (__uint_as_float(0x40000000u) * gradVsZ)) / vs13);
        float termLX = (gradLsqX / vstab);
        float termLY = (gradLsqY / vstab);
        float termLZ = (gradLsqZ / vstab);
        float termVX = ((lsqr * gradVstabX) / sqrVstab);
        float termVY = ((lsqr * gradVstabY) / sqrVstab);
        float termVZ = ((lsqr * gradVstabZ) / sqrVstab);
        float gx = (termLX - termVX);
        float gy = (termLY - termVY);
        float gz = (termLZ - termVZ);
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
