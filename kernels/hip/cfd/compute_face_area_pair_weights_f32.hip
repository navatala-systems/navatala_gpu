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
extern "C" __global__ void navatala_cfd_compute_face_area_pair_weights_f32(const float* sfX, const float* sfY, const float* sfZ, const float* magSf, const unsigned int* nFaces, float* faceWeight) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int f = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFaces[0]));
  if ((f < N)) {
    float sx = sfX[f];
    float sy = sfY[f];
    float sz = sfZ[f];
    float msf = magSf[f];
    float invSqrt = (__uint_as_float(0x3f800000u) / sqrt(msf));
    float wx = (sx * invSqrt);
    float wy = ((sy * __uint_as_float(0x3f8147aeu)) * invSqrt);
    float wz = ((sz * __uint_as_float(0x3f828f5cu)) * invSqrt);
    float mag2 = (((wx * wx) + (wy * wy)) + (wz * wz));
    faceWeight[f] = sqrt(mag2);
  }
}
