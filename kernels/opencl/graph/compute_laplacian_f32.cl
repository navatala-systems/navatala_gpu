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

__kernel void navatala_graph_compute_laplacian_f32(__global const float* W, __global const float* D, __global const uint* n, __global float* L) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      float dVal = D[i];
      L[gid] = dVal;
    } else {
      float wVal = W[gid];
      float negW = (as_float(0x00000000u) - wVal);
      L[gid] = negW;
    }
  }
}
