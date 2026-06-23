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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_spectral_conv4d_mode_sliced(device const float* xFt [[buffer(0)]], device const float* wPPP [[buffer(1)]], device const float* wNPP [[buffer(2)]], device const float* wPNP [[buffer(3)]], device const float* wNNP [[buffer(4)]], device const float* wPPN [[buffer(5)]], device const float* wNPN [[buffer(6)]], device const float* wPNN [[buffer(7)]], device const float* wNNN [[buffer(8)]], device const int* numBatches [[buffer(9)]], device const int* ftT [[buffer(10)]], device const int* ftD [[buffer(11)]], device const int* ftH [[buffer(12)]], device const int* ftW [[buffer(13)]], device float* yFt [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nb = numBatches[0];
  int ftt = ftT[0];
  int ftd = ftD[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 2;
  int m2 = 2;
  int m3 = 2;
  int m4 = 2;
  int hyper = (((ftt * ftd) * fth) * ftw);
  int total = ((nb * outCh) * hyper);
  if (gid < total) {
    int boc = (gid / hyper);
    int tdhwFlat = (gid % hyper);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int t = (tdhwFlat / ((ftd * fth) * ftw));
    int dhwFlat = (tdhwFlat % ((ftd * fth) * ftw));
    int d = (dhwFlat / (fth * ftw));
    int hwFlat = (dhwFlat % (fth * ftw));
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    int negStartT = (ftt - m1);
    int negStartD = (ftd - m2);
    int negStartH = (fth - m3);
    float accR = as_type<float>(0x00000000u);
    float accI = as_type<float>(0x00000000u);
    if ((((t < m1) && (d < m2)) && (h < m3)) && (w < m4)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
        int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + d) * m3) + h) * m4) + w) * 1) + 0);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPPP[((wBase)*2 + 0)];
        float wi = wPPP[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      if ((((t >= negStartT) && (d < m2)) && (h < m3)) && (w < m4)) {
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
          int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + d) * m3) + h) * m4) + w) * 1) + 0);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNPP[((wBase)*2 + 0)];
          float wi = wNPP[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      } else {
        if ((((t < m1) && (d >= negStartD)) && (h < m3)) && (w < m4)) {
          for (int ic = 0; ic < (int)(inCh); ++ic) {
            int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
            int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + (d - negStartD)) * m3) + h) * m4) + w) * 1) + 0);
            float xr = xFt[((xBase)*2 + 0)];
            float xi = xFt[((xBase)*2 + 1)];
            float wr = wPNP[((wBase)*2 + 0)];
            float wi = wPNP[((wBase)*2 + 1)];
            accR = (accR + ((xr * wr) - (xi * wi)));
            accI = (accI + ((xr * wi) + (xi * wr)));
          }
        } else {
          if ((((t >= negStartT) && (d >= negStartD)) && (h < m3)) && (w < m4)) {
            for (int ic = 0; ic < (int)(inCh); ++ic) {
              int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
              int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + (d - negStartD)) * m3) + h) * m4) + w) * 1) + 0);
              float xr = xFt[((xBase)*2 + 0)];
              float xi = xFt[((xBase)*2 + 1)];
              float wr = wNNP[((wBase)*2 + 0)];
              float wi = wNNP[((wBase)*2 + 1)];
              accR = (accR + ((xr * wr) - (xi * wi)));
              accI = (accI + ((xr * wi) + (xi * wr)));
            }
          } else {
            if ((((t < m1) && (d < m2)) && (h >= negStartH)) && (w < m4)) {
              for (int ic = 0; ic < (int)(inCh); ++ic) {
                int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + d) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                float xr = xFt[((xBase)*2 + 0)];
                float xi = xFt[((xBase)*2 + 1)];
                float wr = wPPN[((wBase)*2 + 0)];
                float wi = wPPN[((wBase)*2 + 1)];
                accR = (accR + ((xr * wr) - (xi * wi)));
                accI = (accI + ((xr * wi) + (xi * wr)));
              }
            } else {
              if ((((t >= negStartT) && (d < m2)) && (h >= negStartH)) && (w < m4)) {
                for (int ic = 0; ic < (int)(inCh); ++ic) {
                  int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                  int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + d) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                  float xr = xFt[((xBase)*2 + 0)];
                  float xi = xFt[((xBase)*2 + 1)];
                  float wr = wNPN[((wBase)*2 + 0)];
                  float wi = wNPN[((wBase)*2 + 1)];
                  accR = (accR + ((xr * wr) - (xi * wi)));
                  accI = (accI + ((xr * wi) + (xi * wr)));
                }
              } else {
                if ((((t < m1) && (d >= negStartD)) && (h >= negStartH)) && (w < m4)) {
                  for (int ic = 0; ic < (int)(inCh); ++ic) {
                    int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                    int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + (d - negStartD)) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                    float xr = xFt[((xBase)*2 + 0)];
                    float xi = xFt[((xBase)*2 + 1)];
                    float wr = wPNN[((wBase)*2 + 0)];
                    float wi = wPNN[((wBase)*2 + 1)];
                    accR = (accR + ((xr * wr) - (xi * wi)));
                    accI = (accI + ((xr * wi) + (xi * wr)));
                  }
                } else {
                  if ((((t >= negStartT) && (d >= negStartD)) && (h >= negStartH)) && (w < m4)) {
                    for (int ic = 0; ic < (int)(inCh); ++ic) {
                      int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                      int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + (d - negStartD)) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                      float xr = xFt[((xBase)*2 + 0)];
                      float xi = xFt[((xBase)*2 + 1)];
                      float wr = wNNN[((wBase)*2 + 0)];
                      float wi = wNNN[((wBase)*2 + 1)];
                      accR = (accR + ((xr * wr) - (xi * wi)));
                      accI = (accI + ((xr * wi) + (xi * wr)));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}
