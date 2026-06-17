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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_init_embedding_random_f64(__global const uint* nPoints, __global const uint* outputDims, __global const uint* seed, __global double* Y) {
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
    double maxVal = as_double(0x41dfffffffc00000ul);
    double u1 = (((double)(state1)) / maxVal);
    double u2 = (((double)(state2)) / maxVal);
    double negTwoLnU1 = (as_double(0xc000000000000000ul) * log(u1));
    double sqrtTerm = sqrt(negTwoLnU1);
    double twoPiU2 = (as_double(0x401921fb54411744ul) * u2);
    double z = (sqrtTerm * cos(twoPiU2));
    double yVal = (z * as_double(0x3f847ae147ae147bul));
    Y[gid] = yVal;
  }
}
