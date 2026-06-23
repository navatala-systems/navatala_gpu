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

__kernel void navatala_cfd_primitives_average_face_scalar_to_cell(__global const float* facePhi, __global const float* magSf, __global const uint* offsets, __global const uint* faceIdx, __global const float* sumMag, __global const int* params, __global float* outCell) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    uint begU = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    float acc = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      float term = (magSf[f] * facePhi[f]);
      acc = (acc + term);
    }
    float den = sumMag[((int)((int)(get_global_id(0))))];
    if (den < as_float(0x1e3ce508u)) {
      den = as_float(0x1e3ce508u);
    }
    outCell[((int)((int)(get_global_id(0))))] = (acc / den);
  }
}
