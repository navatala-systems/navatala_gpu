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

__kernel void navatala_ml_compute_equiangular_f32(__global const float* correlations, __global const uint* activeMask, __global const uint* nCols, __global float* equiangular) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == (uint)(1u));
    if (isActive) {
      float corr = correlations[gid];
      bool positive = (corr > as_float(0x00000000u));
      bool negative = (corr < as_float(0x00000000u));
      float signVal = ((positive) ? (as_float(0x3f800000u)) : (((negative) ? (as_float(0xbf800000u)) : (as_float(0x00000000u)))));
      equiangular[gid] = signVal;
    } else {
      equiangular[gid] = as_float(0x00000000u);
    }
  }
}
