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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_compute_attraction_f64(const double* embedding, const int* edge_from, const int* edge_to, const double* edge_weights, const unsigned int* n_edges, const unsigned int* n_components, const double* a_param, const double* b_param, double* gradients) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nEdges = n_edges[0];
  unsigned int nComp = n_components[0];
  double a = a_param[0];
  double b = b_param[0];
  bool inBounds = (gid < nEdges);
  if (inBounds) {
    int fromIdx = edge_from[gid];
    int toIdx = edge_to[gid];
    double weight = edge_weights[gid];
    unsigned int fromIdxU32 = ((unsigned int)(fromIdx));
    unsigned int toIdxU32 = ((unsigned int)(toIdx));
    unsigned int fromBase = (fromIdxU32 * nComp);
    unsigned int toBase = (toIdxU32 * nComp);
    double distSq = __longlong_as_double(0x3f1a36e2eb1c432dull);
    for (int d = 0; d < (int)(nComp); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int fromAddr = (fromBase + dU32);
      unsigned int toAddr = (toBase + dU32);
      double fromVal = embedding[fromAddr];
      double toVal = embedding[toAddr];
      double diff = (fromVal - toVal);
      double diffSq = (diff * diff);
      double currDistSq = distSq;
      double newDistSq = (currDistSq + diffSq);
      distSq = newDistSq;
    }
    double dSq = distSq;
    double bMinus1 = (b - __longlong_as_double(0x3ff0000000000000ull));
    double logDSq = log(dSq);
    double powTerm = exp((bMinus1 * logDSq));
    double dPow2b = exp((b * logDSq));
    double denom = (__longlong_as_double(0x3ff0000000000000ull) + (a * dPow2b));
    double gradScale = ((((__longlong_as_double(0xc000000000000000ull) * a) * b) * (weight * powTerm)) / denom);
    for (int d = 0; d < (int)(nComp); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int fromAddr = (fromBase + dU32);
      unsigned int toAddr = (toBase + dU32);
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
