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

__kernel void navatala_cfd_scalar_ldu_multi_color_gs_sweep(__global const float* diag, __global const float* upper, __global const float* lower, __global const float* rhs, __global const int* owner, __global const int* neighbour, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const uint* colorOrder, __global const uint* colorRange, __global const float* params, __global const uint* counts, __global float* x) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int colorSize = (((int)(colorRange[1])) - ((int)(colorRange[0])));
  if ((int)(get_global_id(0)) >= colorSize) {
    return;
  } else {
    int cellIdx = (((int)(colorRange[0])) + (int)(get_global_id(0)));
    uint celli = colorOrder[cellIdx];
    int ci = ((int)(celli));
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[ci];
    float absD = fabs(d);
    if (absD < smallDiag) {
      if (d >= as_float(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float s = (d * x[ci]);
    int beg = ((int)(offsets[ci]));
    int c1 = (ci + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint f = faceIdx[k];
      int sg = sign[k];
      if (((int)(f)) < ((int)(counts[1]))) {
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    x[ci] = (x[ci] + (omega * ((rhs[ci] - s) / d)));
  }
}
