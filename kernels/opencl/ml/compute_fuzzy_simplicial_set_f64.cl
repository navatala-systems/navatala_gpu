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
__kernel void navatala_ml_compute_fuzzy_simplicial_set_f64(__global const int* knn_indices, __global const double* knn_dists, __global const uint* n_samples, __global const uint* n_neighbors, __global const double* local_connectivity, __global double* membership, __global double* rho, __global double* sigma) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nNeighbors = n_neighbors[0];
  double localConn = local_connectivity[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nNeighbors);
    uint localConnFloor = ((uint)(localConn));
    double localConnFrac = (localConn - ((double)(localConnFloor)));
    uint idxFloor = (sampleBase + localConnFloor);
    uint idxCeil = (idxFloor + (uint)(1u));
    uint maxIdx = ((sampleBase + nNeighbors) - (uint)(1u));
    uint idxCeilClamped = (((idxCeil < maxIdx)) ? (idxCeil) : (maxIdx));
    double distFloor = knn_dists[idxFloor];
    double distCeil = knn_dists[idxCeilClamped];
    double rhoVal = (((as_double(0x3ff0000000000000ul) - localConnFrac) * distFloor) + (localConnFrac * distCeil));
    rho[gid] = rhoVal;
    double nNeighborsF = ((double)(nNeighbors));
    double targetSum = (log(nNeighborsF) / log(as_double(0x4000000000000000ul)));
    double sigmaLow = as_double(0x3ddb7cdfd9d7bdbbul);
    double sigmaHigh = as_double(0x408f400000000000ul);
    double sigmaMid = as_double(0x0000000000000000ul);
    for (int iter = 0; iter < (int)((uint)(64u)); ++iter) {
      double low = sigmaLow;
      double high = sigmaHigh;
      double mid = ((low + high) / as_double(0x4000000000000000ul));
      sigmaMid = mid;
      double membershipSum = as_double(0x0000000000000000ul);
      for (int k = 0; k < (int)(nNeighbors); ++k) {
        uint kU32 = ((uint)(k));
        uint distIdx = (sampleBase + kU32);
        double dist = knn_dists[distIdx];
        double distMinusRho = (dist - rhoVal);
        double negScaled = ((as_double(0x0000000000000000ul) - distMinusRho) / mid);
        double mem = exp(negScaled);
        double memClamped = (((mem > as_double(0x3ff0000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (mem));
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
      uint kU32 = ((uint)(k));
      uint memIdx = (sampleBase + kU32);
      double dist = knn_dists[memIdx];
      double distMinusRho = (dist - rhoVal);
      double negScaled = ((as_double(0x0000000000000000ul) - distMinusRho) / finalSigma);
      double mem = exp(negScaled);
      double memClamped = (((mem > as_double(0x3ff0000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (mem));
      membership[memIdx] = memClamped;
    }
  }
}
