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

__kernel void navatala_cfd_scatter_vec3_and_mask(__global const int* procFaces, __global const float* recvVals, __global const int* counts, __global float* bcX, __global float* bcY, __global float* bcZ, __global uint* bcMask) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int base = ((int)(get_global_id(0)) * 3);
    int b1 = (base + 1);
    int b2 = (base + 2);
    bcX[f] = recvVals[base];
    bcY[f] = recvVals[b1];
    bcZ[f] = recvVals[b2];
    bcMask[f] = ((uint)(1));
  }
}
