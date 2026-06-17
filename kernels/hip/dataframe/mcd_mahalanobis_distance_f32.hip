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
extern "C" __global__ void navatala_dataframe_mcd_mahalanobis_distance_f32(const float* dataX, const float* dataY, const float* robustMeanX, const float* robustMeanY, const float* robustCov, const float* varX, const float* varY, const unsigned int* n, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    float mx = robustMeanX[0];
    float my = robustMeanY[0];
    float cov = robustCov[0];
    float vx = varX[0];
    float vy = varY[0];
    float x = dataX[gid];
    float y = dataY[gid];
    float dx = (x - mx);
    float dy = (y - my);
    float covSq = (cov * cov);
    float det = ((vx * vy) - covSq);
    float safeDet = (((det > __uint_as_float(0x358637bdu))) ? (det) : (__uint_as_float(0x3f800000u)));
    float invDet = (__uint_as_float(0x3f800000u) / safeDet);
    float dx2 = (dx * dx);
    float dy2 = (dy * dy);
    float dxdy = (dx * dy);
    float term1 = (dx2 * vy);
    float term2 = (dy2 * vx);
    float term3 = (__uint_as_float(0x40000000u) * (dxdy * cov));
    float num = ((term1 + term2) - term3);
    float dist2 = (num * invDet);
    distances[gid] = dist2;
  }
}
