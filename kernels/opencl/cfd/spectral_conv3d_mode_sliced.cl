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

__kernel void navatala_cfd_spectral_conv3d_mode_sliced(__global const float* xFt, __global const float* wPP, __global const float* wNP, __global const float* wPN, __global const float* wNN, __global const int* numBatches, __global const int* ftD, __global const int* ftH, __global const int* ftW, __global float* yFt) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nb = numBatches[0];
  int ftd = ftD[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 4;
  int m2 = 4;
  int m3 = 4;
  int plane = ((ftd * fth) * ftw);
  int total = ((nb * outCh) * plane);
  if (gid < total) {
    int boc = (gid / plane);
    int dhwFlat = (gid % plane);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int d = (dhwFlat / (fth * ftw));
    int hwFlat = (dhwFlat % (fth * ftw));
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    int negStartD = (ftd - m1);
    int negStartH = (fth - m2);
    float accR = as_float(0x00000000u);
    float accI = as_float(0x00000000u);
    if (((d < m1) && (h < m2)) && (w < m3)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
        int wBase = ((((((((oc * inCh) + ic) * m1) + d) * m2) + h) * m3) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPP[((wBase)*2 + 0)];
        float wi = wPP[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      if (((d >= negStartD) && (h < m2)) && (w < m3)) {
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
          int wBase = ((((((((oc * inCh) + ic) * m1) + (d - negStartD)) * m2) + h) * m3) + w);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNP[((wBase)*2 + 0)];
          float wi = wNP[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      } else {
        if (((d < m1) && (h >= negStartH)) && (w < m3)) {
          for (int ic = 0; ic < (int)(inCh); ++ic) {
            int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
            int wBase = ((((((((oc * inCh) + ic) * m1) + d) * m2) + (h - negStartH)) * m3) + w);
            float xr = xFt[((xBase)*2 + 0)];
            float xi = xFt[((xBase)*2 + 1)];
            float wr = wPN[((wBase)*2 + 0)];
            float wi = wPN[((wBase)*2 + 1)];
            accR = (accR + ((xr * wr) - (xi * wi)));
            accI = (accI + ((xr * wi) + (xi * wr)));
          }
        } else {
          if (((d >= negStartD) && (h >= negStartH)) && (w < m3)) {
            for (int ic = 0; ic < (int)(inCh); ++ic) {
              int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
              int wBase = ((((((((oc * inCh) + ic) * m1) + (d - negStartD)) * m2) + (h - negStartH)) * m3) + w);
              float xr = xFt[((xBase)*2 + 0)];
              float xi = xFt[((xBase)*2 + 1)];
              float wr = wNN[((wBase)*2 + 0)];
              float wi = wNN[((wBase)*2 + 1)];
              accR = (accR + ((xr * wr) - (xi * wi)));
              accI = (accI + ((xr * wi) + (xi * wr)));
            }
          }
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}
