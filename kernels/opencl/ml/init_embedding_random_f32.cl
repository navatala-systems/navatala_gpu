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

__kernel void navatala_ml_init_embedding_random_f32(__global const uint* nPoints, __global const uint* outputDims, __global const uint* seed, __global float* Y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint s = seed[0];
  uint totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint state = (s + gid);
    uint lcgA = (uint)(48271u);
    uint state1 = (lcgA * state);
    uint state2 = (lcgA * state1);
    float maxVal = as_float(0x4f000000u);
    float u1 = (((float)(state1)) / maxVal);
    float u2 = (((float)(state2)) / maxVal);
    float negTwoLnU1 = (as_float(0xc0000000u) * log(u1));
    float sqrtTerm = sqrt(negTwoLnU1);
    float twoPiU2 = (as_float(0x40c90fdbu) * u2);
    float z = (sqrtTerm * cos(twoPiU2));
    float yVal = (z * as_float(0x3c23d70au));
    Y[gid] = yVal;
  }
}
