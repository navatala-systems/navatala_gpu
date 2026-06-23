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

kernel void navatala_cfd_spectral_conv1d_mode_sliced(device const float* xFt [[buffer(0)]], device const float* wPos [[buffer(1)]], device const float* wNeg [[buffer(2)]], device const int* numBatches [[buffer(3)]], device const int* ftW [[buffer(4)]], device float* yFt [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nb = numBatches[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 8;
  int total = ((nb * outCh) * ftw);
  if (gid < total) {
    int boc = (gid / ftw);
    int w = (gid % ftw);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    float accR = as_type<float>(0x00000000u);
    float accI = as_type<float>(0x00000000u);
    if (w < m1) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((b * inCh) + ic) * ftw) + w);
        int wBase = ((((oc * inCh) + ic) * m1) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPos[((wBase)*2 + 0)];
        float wi = wPos[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      int negStart = (ftw - m1);
      if (w >= negStart) {
        int wLocal = (w - negStart);
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((b * inCh) + ic) * ftw) + w);
          int wBase = ((((oc * inCh) + ic) * m1) + wLocal);
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
