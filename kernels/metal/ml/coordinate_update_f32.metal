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

kernel void navatala_ml_coordinate_update_f32(device const float* X [[buffer(0)]], device const float* residuals [[buffer(1)]], device const uint* j [[buffer(2)]], device const float* lambda [[buffer(3)]], device const uint* nSamples [[buffer(4)]], device const uint* nFeatures [[buffer(5)]], device float* newBetaJ [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdataDot[256];
  threadgroup float sdataNorm[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint xijIdx = ((gid * d) + jVal);
    float xij = X[xijIdx];
    float ri = residuals[gid];
    float dotProd = (xij * ri);
    float xijSq = (xij * xij);
    sdataDot[lid] = dotProd;
    sdataNorm[lid] = xijSq;
  } else {
    sdataDot[lid] = as_type<float>(0x00000000u);
    sdataNorm[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cu32RedStride = 128u;
  for (int cu32RedStep = 0; cu32RedStep < (int)(8); ++cu32RedStep) {
    uint cu32RedStrideVal = cu32RedStride;
    if (lid < cu32RedStrideVal) {
      float otherDot = sdataDot[(lid + cu32RedStrideVal)];
      float mineDot = sdataDot[lid];
      float sumDot = (mineDot + otherDot);
      sdataDot[lid] = sumDot;
      float otherNorm = sdataNorm[(lid + cu32RedStrideVal)];
      float mineNorm = sdataNorm[lid];
      float sumNorm = (mineNorm + otherNorm);
      sdataNorm[lid] = sumNorm;
    }
    uint cu32RedStrideToHalve = cu32RedStride;
    uint cu32RedNextStride = (cu32RedStrideToHalve >> 1u);
    cu32RedStride = cu32RedNextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalDot = sdataDot[0];
    float totalNorm = sdataNorm[0];
    float rho = (totalDot / totalNorm);
    float thresh = (lambdaVal / totalNorm);
    float absRho = abs(rho);
    float diff = (absRho - thresh);
    float maxDiff = (((diff > as_type<float>(0x00000000u))) ? (diff) : (as_type<float>(0x00000000u)));
    bool positive = (rho > as_type<float>(0x00000000u));
    bool negative = (rho < as_type<float>(0x00000000u));
    float signVal = ((positive) ? (as_type<float>(0x3f800000u)) : (((negative) ? (as_type<float>(0xbf800000u)) : (as_type<float>(0x00000000u)))));
    float newBeta = (signVal * maxDiff);
    newBetaJ[0] = newBeta;
  }
}
