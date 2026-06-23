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

__kernel void navatala_cfd_scatter_mu_grad_ghost(__global const float* recvVals, __global const int* counts, __global float* muCell, __global float* gXX, __global float* gXY, __global float* gXZ, __global float* gYX, __global float* gYY, __global float* gYZ, __global float* gZX, __global float* gZY, __global float* gZZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int idx = (((int)(counts[1])) + (int)(get_global_id(0)));
    int base = ((int)(get_global_id(0)) * 10);
    muCell[idx] = recvVals[base];
    int b1 = (base + 1);
    gXX[idx] = recvVals[b1];
    int b2 = (base + 2);
    gXY[idx] = recvVals[b2];
    int b3 = (base + 3);
    gXZ[idx] = recvVals[b3];
    int b4 = (base + 4);
    gYX[idx] = recvVals[b4];
    int b5 = (base + 5);
    gYY[idx] = recvVals[b5];
    int b6 = (base + 6);
    gYZ[idx] = recvVals[b6];
    int b7 = (base + 7);
    gZX[idx] = recvVals[b7];
    int b8 = (base + 8);
    gZY[idx] = recvVals[b8];
    int b9 = (base + 9);
    gZZ[idx] = recvVals[b9];
  }
}
