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

__kernel void navatala_ml_compute_fuzzy_simplicial_set_f32(__global const int* knn_indices, __global const float* knn_dists, __global const uint* n_samples, __global const uint* n_neighbors, __global const float* local_connectivity, __global float* membership, __global float* rho, __global float* sigma) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nNeighbors = n_neighbors[0];
  float localConn = local_connectivity[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nNeighbors);
    uint localConnFloor = ((uint)(localConn));
    float localConnFrac = (localConn - ((float)(localConnFloor)));
    uint idxFloor = (sampleBase + localConnFloor);
    uint idxCeil = (idxFloor + (uint)(1u));
    uint maxIdx = ((sampleBase + nNeighbors) - (uint)(1u));
    uint idxCeilClamped = (((idxCeil < maxIdx)) ? (idxCeil) : (maxIdx));
    float distFloor = knn_dists[idxFloor];
    float distCeil = knn_dists[idxCeilClamped];
    float rhoVal = (((as_float(0x3f800000u) - localConnFrac) * distFloor) + (localConnFrac * distCeil));
    rho[gid] = rhoVal;
    float nNeighborsF = ((float)(nNeighbors));
    float targetSum = (log(nNeighborsF) / log(as_float(0x40000000u)));
    float sigmaLow = as_float(0x2edbe6ffu);
    float sigmaHigh = as_float(0x447a0000u);
    float sigmaMid = as_float(0x00000000u);
    for (int iter = 0; iter < (int)((uint)(64u)); ++iter) {
      float low = sigmaLow;
      float high = sigmaHigh;
      float mid = ((low + high) / as_float(0x40000000u));
      sigmaMid = mid;
      float membershipSum = as_float(0x00000000u);
      for (int k = 0; k < (int)(nNeighbors); ++k) {
        uint kU32 = ((uint)(k));
        uint distIdx = (sampleBase + kU32);
        float dist = knn_dists[distIdx];
        float distMinusRho = (dist - rhoVal);
        float negScaled = ((as_float(0x00000000u) - distMinusRho) / mid);
        float mem = exp(negScaled);
        float memClamped = (((mem > as_float(0x3f800000u))) ? (as_float(0x3f800000u)) : (mem));
        float currSum = membershipSum;
        float newSum = (currSum + memClamped);
        membershipSum = newSum;
      }
      float sum = membershipSum;
      if ((sum > targetSum)) {
        sigmaHigh = mid;
      } else {
        sigmaLow = mid;
      }
    }
    float finalSigma = sigmaMid;
    sigma[gid] = finalSigma;
    for (int k = 0; k < (int)(nNeighbors); ++k) {
      uint kU32 = ((uint)(k));
      uint memIdx = (sampleBase + kU32);
      float dist = knn_dists[memIdx];
      float distMinusRho = (dist - rhoVal);
      float negScaled = ((as_float(0x00000000u) - distMinusRho) / finalSigma);
      float mem = exp(negScaled);
      float memClamped = (((mem > as_float(0x3f800000u))) ? (as_float(0x3f800000u)) : (mem));
      membership[memIdx] = memClamped;
    }
  }
}
