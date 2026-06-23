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

__kernel void navatala_cfd_vof_average_face_scalar_to_cell(__global const float* alphaF, __global const float* magSf, __global const uint* offsets, __global const uint* faceIdx, __global const float* sumMag, __global const uint* counts, __global float* outAlpha) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = as_float(0x00000000u);
    uint beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      s = (s + (magSf[((int)(f))] * alphaF[((int)(f))]));
    }
    float d = sumMag[(int)(get_global_id(0))];
    if (d < as_float(0x1e3ce508u)) {
      outAlpha[(int)(get_global_id(0))] = (s / as_float(0x1e3ce508u));
    } else {
      outAlpha[(int)(get_global_id(0))] = (s / d);
    }
  }
}
