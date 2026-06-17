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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_fuzzy_simplicial_set_f32(device const int* knn_indices [[buffer(0)]], device const float* knn_dists [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device const uint* n_neighbors [[buffer(3)]], device const float* local_connectivity [[buffer(4)]], device float* membership [[buffer(5)]], device float* rho [[buffer(6)]], device float* sigma [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nNeighbors = n_neighbors[0];
  float localConn = local_connectivity[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nNeighbors);
    uint localConnFloor = ((uint)(localConn));
    float localConnFrac = (localConn - ((float)(localConnFloor)));
    uint idxFloor = (sampleBase + localConnFloor);
    uint idxCeil = (idxFloor + 1u);
    uint maxIdx = ((sampleBase + nNeighbors) - 1u);
    uint idxCeilClamped = (((idxCeil < maxIdx)) ? (idxCeil) : (maxIdx));
    float distFloor = knn_dists[idxFloor];
    float distCeil = knn_dists[idxCeilClamped];
    float rhoVal = (((as_type<float>(0x3f800000u) - localConnFrac) * distFloor) + (localConnFrac * distCeil));
    rho[gid] = rhoVal;
    float nNeighborsF = ((float)(nNeighbors));
    float targetSum = (log(nNeighborsF) / log(as_type<float>(0x40000000u)));
    float sigmaLow = as_type<float>(0x2edbe6ffu);
    float sigmaHigh = as_type<float>(0x447a0000u);
    float sigmaMid = as_type<float>(0x00000000u);
    for (int iter = 0; iter < (int)(64u); ++iter) {
      float low = sigmaLow;
      float high = sigmaHigh;
      float mid = ((low + high) / as_type<float>(0x40000000u));
      sigmaMid = mid;
      float membershipSum = as_type<float>(0x00000000u);
      for (int k = 0; k < (int)(nNeighbors); ++k) {
        uint kU32 = ((uint)(k));
        uint distIdx = (sampleBase + kU32);
        float dist = knn_dists[distIdx];
        float distMinusRho = (dist - rhoVal);
        float negScaled = ((as_type<float>(0x00000000u) - distMinusRho) / mid);
        float mem = exp(negScaled);
        float memClamped = (((mem > as_type<float>(0x3f800000u))) ? (as_type<float>(0x3f800000u)) : (mem));
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
      float negScaled = ((as_type<float>(0x00000000u) - distMinusRho) / finalSigma);
      float mem = exp(negScaled);
      float memClamped = (((mem > as_type<float>(0x3f800000u))) ? (as_type<float>(0x3f800000u)) : (mem));
      membership[memIdx] = memClamped;
    }
  }
}
