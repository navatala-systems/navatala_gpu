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
extern "C" __global__ void navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2(const float* muCell, const float* muBc, const float* gXX, const float* gXY, const float* gXZ, const float* gYX, const float* gYY, const float* gYZ, const float* gZX, const float* gZY, const float* gZZ, const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* signF, const float* vol, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float sumX = __uint_as_float(0x00000000u);
    float sumY = __uint_as_float(0x00000000u);
    float sumZ = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        float s = signF[k];
        int o = owner[f];
        float muO = muCell[o];
        float goXX = gXX[o];
        float goXY = gXY[o];
        float goXZ = gXZ[o];
        float goYX = gYX[o];
        float goYY = gYY[o];
        float goYZ = gYZ[o];
        float goZX = gZX[o];
        float goZY = gZY[o];
        float goZZ = gZZ[o];
        float trO = (goXX + (goYY + goZZ));
        float twoThirdO = (__uint_as_float(0x3f2aaaabu) * trO);
        float tOxx = (goXX - twoThirdO);
        float tOxy = goYX;
        float tOxz = goZX;
        float tOyx = goXY;
        float tOyy = (goYY - twoThirdO);
        float tOyz = goZY;
        float tOzx = goXZ;
        float tOzy = goYZ;
        float tOzz = (goZZ - twoThirdO);
        float tauOxx = (muO * tOxx);
        float tauOxy = (muO * tOxy);
        float tauOxz = (muO * tOxz);
        float tauOyx = (muO * tOyx);
        float tauOyy = (muO * tOyy);
        float tauOyz = (muO * tOyz);
        float tauOzx = (muO * tOzx);
        float tauOzy = (muO * tOzy);
        float tauOzz = (muO * tOzz);
        float tauFxx = tauOxx;
        float tauFxy = tauOxy;
        float tauFxz = tauOxz;
        float tauFyx = tauOyx;
        float tauFyy = tauOyy;
        float tauFyz = tauOyz;
        float tauFzx = tauOzx;
        float tauFzy = tauOzy;
        float tauFzz = tauOzz;
        int n = neighbour[f];
        if (n >= 0) {
          float muN = muCell[n];
          float gnXX = gXX[n];
          float gnXY = gXY[n];
          float gnXZ = gXZ[n];
          float gnYX = gYX[n];
          float gnYY = gYY[n];
          float gnYZ = gYZ[n];
          float gnZX = gZX[n];
          float gnZY = gZY[n];
          float gnZZ = gZZ[n];
          float trN = (gnXX + (gnYY + gnZZ));
          float twoThirdN = (__uint_as_float(0x3f2aaaabu) * trN);
          float tNxx = (gnXX - twoThirdN);
          float tNxy = gnYX;
          float tNxz = gnZX;
          float tNyx = gnXY;
          float tNyy = (gnYY - twoThirdN);
          float tNyz = gnZY;
          float tNzx = gnXZ;
          float tNzy = gnYZ;
          float tNzz = (gnZZ - twoThirdN);
          float tauNxx = (muN * tNxx);
          float tauNxy = (muN * tNxy);
          float tauNxz = (muN * tNxz);
          float tauNyx = (muN * tNyx);
          float tauNyy = (muN * tNyy);
          float tauNyz = (muN * tNyz);
          float tauNzx = (muN * tNzx);
          float tauNzy = (muN * tNzy);
          float tauNzz = (muN * tNzz);
          float ww = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - ww);
          tauFxx = ((ww * tauOxx) + (iw * tauNxx));
          tauFxy = ((ww * tauOxy) + (iw * tauNxy));
          tauFxz = ((ww * tauOxz) + (iw * tauNxz));
          tauFyx = ((ww * tauOyx) + (iw * tauNyx));
          tauFyy = ((ww * tauOyy) + (iw * tauNyy));
          tauFyz = ((ww * tauOyz) + (iw * tauNyz));
          tauFzx = ((ww * tauOzx) + (iw * tauNzx));
          tauFzy = ((ww * tauOzy) + (iw * tauNzy));
          tauFzz = ((ww * tauOzz) + (iw * tauNzz));
        }
        float sx = sfX[f];
        float sy = sfY[f];
        float sz = sfZ[f];
        float fluxX = ((sx * tauFxx) + ((sy * tauFyx) + (sz * tauFzx)));
        float fluxY = ((sx * tauFxy) + ((sy * tauFyy) + (sz * tauFzy)));
        float fluxZ = ((sx * tauFxz) + ((sy * tauFyz) + (sz * tauFzz)));
        float wX = (s * fluxX);
        float wY = (s * fluxY);
        float wZ = (s * fluxZ);
        sumX = (sumX + wX);
        sumY = (sumY + wY);
        sumZ = (sumZ + wZ);
      }
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumX * invV);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumY * invV);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZ * invV);
  }
}
