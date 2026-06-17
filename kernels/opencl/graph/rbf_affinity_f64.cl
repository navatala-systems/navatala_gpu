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
__kernel void navatala_graph_rbf_affinity_f64(__global const double* data, __global const double* sigma, __global const uint* n, __global const uint* d, __global double* W) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint dVal = d[0];
  double sigmaVal = sigma[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = as_double(0x0000000000000000ul);
    } else {
      double distSq = as_double(0x0000000000000000ul);
      uint rowI = (i * dVal);
      uint rowJ = (j * dVal);
      uint idxI = rowI;
      uint idxJ = rowJ;
      double xi = data[idxI];
      double xj = data[idxJ];
      double diff = (xi - xj);
      double diffSq = (diff * diff);
      double sigmaSq = (sigmaVal * sigmaVal);
      double twoSigmaSq = (as_double(0x4000000000000000ul) * sigmaSq);
      double exponent = ((as_double(0x0000000000000000ul) - diffSq) / twoSigmaSq);
      double affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}
