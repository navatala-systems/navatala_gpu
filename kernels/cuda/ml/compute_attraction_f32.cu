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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_compute_attraction_f32(const float* embedding, const int* edge_from, const int* edge_to, const float* edge_weights, const unsigned int* n_edges, const unsigned int* n_components, const float* a_param, const float* b_param, float* gradients) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nEdges = n_edges[0];
  unsigned int nComp = n_components[0];
  float a = a_param[0];
  float b = b_param[0];
  bool inBounds = (gid < nEdges);
  if (inBounds) {
    int fromIdx = edge_from[gid];
    int toIdx = edge_to[gid];
    float weight = edge_weights[gid];
    unsigned int fromIdxU32 = ((unsigned int)(fromIdx));
    unsigned int toIdxU32 = ((unsigned int)(toIdx));
    unsigned int fromBase = (fromIdxU32 * nComp);
    unsigned int toBase = (toIdxU32 * nComp);
    float distSq = __uint_as_float(0x38d1b717u);
    for (int d = 0; d < (int)(nComp); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int fromAddr = (fromBase + dU32);
      unsigned int toAddr = (toBase + dU32);
      float fromVal = embedding[fromAddr];
      float toVal = embedding[toAddr];
      float diff = (fromVal - toVal);
      float diffSq = (diff * diff);
      float currDistSq = distSq;
      float newDistSq = (currDistSq + diffSq);
      distSq = newDistSq;
    }
    float dSq = distSq;
    float bMinus1 = (b - __uint_as_float(0x3f800000u));
    float logDSq = log(dSq);
    float powTerm = exp((bMinus1 * logDSq));
    float dPow2b = exp((b * logDSq));
    float denom = (__uint_as_float(0x3f800000u) + (a * dPow2b));
    float gradScale = ((((__uint_as_float(0xc0000000u) * a) * b) * (weight * powTerm)) / denom);
    for (int d = 0; d < (int)(nComp); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int fromAddr = (fromBase + dU32);
      unsigned int toAddr = (toBase + dU32);
      float fromVal = embedding[fromAddr];
      float toVal = embedding[toAddr];
      float diff = (fromVal - toVal);
      float grad = (gradScale * diff);
      float currGrad = gradients[fromAddr];
      float newGrad = (currGrad + grad);
      gradients[fromAddr] = newGrad;
    }
  }
}
