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

__kernel void navatala_cfd_spectral_conv2d_mode_sliced(__global const float* xFt, __global const float* wPos, __global const float* wNeg, __global const int* numBatches, __global const int* ftH, __global const int* ftW, __global float* yFt) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nb = numBatches[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 8;
  int m2 = 8;
  int plane = (fth * ftw);
  int total = ((nb * outCh) * plane);
  if (gid < total) {
    int boc = (gid / plane);
    int hwFlat = (gid % plane);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    float accR = as_float(0x00000000u);
    float accI = as_float(0x00000000u);
    if ((h < m1) && (w < m2)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((b * inCh) + ic) * fth) + h) * ftw) + w);
        int wBase = ((((((oc * inCh) + ic) * m1) + h) * m2) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPos[((wBase)*2 + 0)];
        float wi = wPos[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      int negStart = (fth - m1);
      if ((h >= negStart) && (w < m2)) {
        int hLocal = (h - negStart);
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((b * inCh) + ic) * fth) + h) * ftw) + w);
          int wBase = ((((((oc * inCh) + ic) * m1) + hLocal) * m2) + w);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNeg[((wBase)*2 + 0)];
          float wi = wNeg[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}
