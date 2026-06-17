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

__kernel void navatala_cfd_pack_mu_grad_owner(__global const float* muCell, __global const float* gXX, __global const float* gXY, __global const float* gXZ, __global const float* gYX, __global const float* gYY, __global const float* gYZ, __global const float* gZX, __global const float* gZY, __global const float* gZZ, __global const int* owner, __global const int* procFaces, __global const int* counts, __global float* outSend) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int o = owner[f];
    int base = ((int)(get_global_id(0)) * 10);
    outSend[base] = muCell[o];
    int b1 = (base + 1);
    outSend[b1] = gXX[o];
    int b2 = (base + 2);
    outSend[b2] = gXY[o];
    int b3 = (base + 3);
    outSend[b3] = gXZ[o];
    int b4 = (base + 4);
    outSend[b4] = gYX[o];
    int b5 = (base + 5);
    outSend[b5] = gYY[o];
    int b6 = (base + 6);
    outSend[b6] = gYZ[o];
    int b7 = (base + 7);
    outSend[b7] = gZX[o];
    int b8 = (base + 8);
    outSend[b8] = gZY[o];
    int b9 = (base + 9);
    outSend[b9] = gZZ[o];
  }
}
