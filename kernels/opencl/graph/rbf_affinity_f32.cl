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

__kernel void navatala_graph_rbf_affinity_f32(__global const float* data, __global const float* sigma, __global const uint* n, __global const uint* d, __global float* W) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint dVal = d[0];
  float sigmaVal = sigma[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = as_float(0x00000000u);
    } else {
      float distSq = as_float(0x00000000u);
      uint rowI = (i * dVal);
      uint rowJ = (j * dVal);
      uint idxI = rowI;
      uint idxJ = rowJ;
      float xi = data[idxI];
      float xj = data[idxJ];
      float diff = (xi - xj);
      float diffSq = (diff * diff);
      float sigmaSq = (sigmaVal * sigmaVal);
      float twoSigmaSq = (as_float(0x40000000u) * sigmaSq);
      float exponent = ((as_float(0x00000000u) - diffSq) / twoSigmaSq);
      float affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}
