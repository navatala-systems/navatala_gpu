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
extern "C" __global__ void navatala_ml_compute_fuzzy_simplicial_set_f64(const int* knn_indices, const double* knn_dists, const unsigned int* n_samples, const unsigned int* n_neighbors, const double* local_connectivity, double* membership, double* rho, double* sigma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nNeighbors = n_neighbors[0];
  double localConn = local_connectivity[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nNeighbors);
    unsigned int localConnFloor = ((unsigned int)(localConn));
    double localConnFrac = (localConn - ((double)(localConnFloor)));
    unsigned int idxFloor = (sampleBase + localConnFloor);
    unsigned int idxCeil = (idxFloor + 1u);
    unsigned int maxIdx = ((sampleBase + nNeighbors) - 1u);
    unsigned int idxCeilClamped = (((idxCeil < maxIdx)) ? (idxCeil) : (maxIdx));
    double distFloor = knn_dists[idxFloor];
    double distCeil = knn_dists[idxCeilClamped];
    double rhoVal = (((__longlong_as_double(0x3ff0000000000000ull) - localConnFrac) * distFloor) + (localConnFrac * distCeil));
    rho[gid] = rhoVal;
    double nNeighborsF = ((double)(nNeighbors));
    double targetSum = (log(nNeighborsF) / log(__longlong_as_double(0x4000000000000000ull)));
    double sigmaLow = __longlong_as_double(0x3ddb7cdfd9d7bdbbull);
    double sigmaHigh = __longlong_as_double(0x408f400000000000ull);
    double sigmaMid = __longlong_as_double(0x0000000000000000ull);
    for (int iter = 0; iter < (int)(64u); ++iter) {
      double low = sigmaLow;
      double high = sigmaHigh;
      double mid = ((low + high) / __longlong_as_double(0x4000000000000000ull));
      sigmaMid = mid;
      double membershipSum = __longlong_as_double(0x0000000000000000ull);
      for (int k = 0; k < (int)(nNeighbors); ++k) {
        unsigned int kU32 = ((unsigned int)(k));
        unsigned int distIdx = (sampleBase + kU32);
        double dist = knn_dists[distIdx];
        double distMinusRho = (dist - rhoVal);
        double negScaled = ((__longlong_as_double(0x0000000000000000ull) - distMinusRho) / mid);
        double mem = exp(negScaled);
        double memClamped = (((mem > __longlong_as_double(0x3ff0000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (mem));
        double currSum = membershipSum;
        double newSum = (currSum + memClamped);
        membershipSum = newSum;
      }
      double sum = membershipSum;
      if ((sum > targetSum)) {
        sigmaHigh = mid;
      } else {
        sigmaLow = mid;
      }
    }
    double finalSigma = sigmaMid;
    sigma[gid] = finalSigma;
    for (int k = 0; k < (int)(nNeighbors); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int memIdx = (sampleBase + kU32);
      double dist = knn_dists[memIdx];
      double distMinusRho = (dist - rhoVal);
      double negScaled = ((__longlong_as_double(0x0000000000000000ull) - distMinusRho) / finalSigma);
      double mem = exp(negScaled);
      double memClamped = (((mem > __longlong_as_double(0x3ff0000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (mem));
      membership[memIdx] = memClamped;
    }
  }
}
