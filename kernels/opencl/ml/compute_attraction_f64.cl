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
__kernel void navatala_ml_compute_attraction_f64(__global const double* embedding, __global const int* edge_from, __global const int* edge_to, __global const double* edge_weights, __global const uint* n_edges, __global const uint* n_components, __global const double* a_param, __global const double* b_param, __global double* gradients) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nEdges = n_edges[0];
  uint nComp = n_components[0];
  double a = a_param[0];
  double b = b_param[0];
  bool inBounds = (gid < nEdges);
  if (inBounds) {
    int fromIdx = edge_from[gid];
    int toIdx = edge_to[gid];
    double weight = edge_weights[gid];
    uint fromIdxU32 = ((uint)(fromIdx));
    uint toIdxU32 = ((uint)(toIdx));
    uint fromBase = (fromIdxU32 * nComp);
    uint toBase = (toIdxU32 * nComp);
    double distSq = as_double(0x3f1a36e2eb1c432dul);
    for (int d = 0; d < (int)(nComp); ++d) {
      uint dU32 = ((uint)(d));
      uint fromAddr = (fromBase + dU32);
      uint toAddr = (toBase + dU32);
      double fromVal = embedding[fromAddr];
      double toVal = embedding[toAddr];
      double diff = (fromVal - toVal);
      double diffSq = (diff * diff);
      double currDistSq = distSq;
      double newDistSq = (currDistSq + diffSq);
      distSq = newDistSq;
    }
    double dSq = distSq;
    double bMinus1 = (b - as_double(0x3ff0000000000000ul));
    double logDSq = log(dSq);
    double powTerm = exp((bMinus1 * logDSq));
    double dPow2b = exp((b * logDSq));
    double denom = (as_double(0x3ff0000000000000ul) + (a * dPow2b));
    double gradScale = ((((as_double(0xc000000000000000ul) * a) * b) * (weight * powTerm)) / denom);
    for (int d = 0; d < (int)(nComp); ++d) {
      uint dU32 = ((uint)(d));
      uint fromAddr = (fromBase + dU32);
      uint toAddr = (toBase + dU32);
      double fromVal = embedding[fromAddr];
      double toVal = embedding[toAddr];
      double diff = (fromVal - toVal);
      double grad = (gradScale * diff);
      double currGrad = gradients[fromAddr];
      double newGrad = (currGrad + grad);
      gradients[fromAddr] = newGrad;
    }
  }
}
