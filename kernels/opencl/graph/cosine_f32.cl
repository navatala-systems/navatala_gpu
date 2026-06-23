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

__kernel void navatala_graph_cosine_f32(__global const uint* inter, __global const uint* degrees, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global float* coeff) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((sqrt((dAF * dBF)) > as_float(0x00000000u))) ? ((iF / sqrt((dAF * dBF)))) : (as_float(0x00000000u)));
  }
}
