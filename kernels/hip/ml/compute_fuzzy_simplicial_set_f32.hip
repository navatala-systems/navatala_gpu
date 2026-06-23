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
extern "C" __global__ void navatala_ml_compute_fuzzy_simplicial_set_f32(const int* knn_indices, const float* knn_dists, const unsigned int* n_samples, const unsigned int* n_neighbors, const float* local_connectivity, float* membership, float* rho, float* sigma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nNeighbors = n_neighbors[0];
  float localConn = local_connectivity[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nNeighbors);
    unsigned int localConnFloor = ((unsigned int)(localConn));
    float localConnFrac = (localConn - ((float)(localConnFloor)));
    unsigned int idxFloor = (sampleBase + localConnFloor);
    unsigned int idxCeil = (idxFloor + 1u);
    unsigned int maxIdx = ((sampleBase + nNeighbors) - 1u);
    unsigned int idxCeilClamped = (((idxCeil < maxIdx)) ? (idxCeil) : (maxIdx));
    float distFloor = knn_dists[idxFloor];
    float distCeil = knn_dists[idxCeilClamped];
    float rhoVal = (((__uint_as_float(0x3f800000u) - localConnFrac) * distFloor) + (localConnFrac * distCeil));
    rho[gid] = rhoVal;
    float nNeighborsF = ((float)(nNeighbors));
    float targetSum = (log(nNeighborsF) / log(__uint_as_float(0x40000000u)));
    float sigmaLow = __uint_as_float(0x2edbe6ffu);
    float sigmaHigh = __uint_as_float(0x447a0000u);
    float sigmaMid = __uint_as_float(0x00000000u);
    for (int iter = 0; iter < (int)(64u); ++iter) {
      float low = sigmaLow;
      float high = sigmaHigh;
      float mid = ((low + high) / __uint_as_float(0x40000000u));
      sigmaMid = mid;
      float membershipSum = __uint_as_float(0x00000000u);
      for (int k = 0; k < (int)(nNeighbors); ++k) {
        unsigned int kU32 = ((unsigned int)(k));
        unsigned int distIdx = (sampleBase + kU32);
        float dist = knn_dists[distIdx];
        float distMinusRho = (dist - rhoVal);
        float negScaled = ((__uint_as_float(0x00000000u) - distMinusRho) / mid);
        float mem = exp(negScaled);
        float memClamped = (((mem > __uint_as_float(0x3f800000u))) ? (__uint_as_float(0x3f800000u)) : (mem));
        float currSum = membershipSum;
        float newSum = (currSum + memClamped);
        membershipSum = newSum;
      }
      float sum = membershipSum;
      if (sum > targetSum) {
        sigmaHigh = mid;
      } else {
        sigmaLow = mid;
      }
    }
    float finalSigma = sigmaMid;
    sigma[gid] = finalSigma;
    for (int k = 0; k < (int)(nNeighbors); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int memIdx = (sampleBase + kU32);
      float dist = knn_dists[memIdx];
      float distMinusRho = (dist - rhoVal);
      float negScaled = ((__uint_as_float(0x00000000u) - distMinusRho) / finalSigma);
      float mem = exp(negScaled);
      float memClamped = (((mem > __uint_as_float(0x3f800000u))) ? (__uint_as_float(0x3f800000u)) : (mem));
      membership[memIdx] = memClamped;
    }
  }
}
