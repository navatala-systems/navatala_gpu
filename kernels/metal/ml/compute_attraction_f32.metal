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

kernel void navatala_ml_compute_attraction_f32(device const float* embedding [[buffer(0)]], device const int* edge_from [[buffer(1)]], device const int* edge_to [[buffer(2)]], device const float* edge_weights [[buffer(3)]], device const uint* n_edges [[buffer(4)]], device const uint* n_components [[buffer(5)]], device const float* a_param [[buffer(6)]], device const float* b_param [[buffer(7)]], device float* gradients [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nEdges = n_edges[0];
  uint nComp = n_components[0];
  float a = a_param[0];
  float b = b_param[0];
  bool inBounds = (gid < nEdges);
  if (inBounds) {
    int fromIdx = edge_from[gid];
    int toIdx = edge_to[gid];
    float weight = edge_weights[gid];
    uint fromIdxU32 = ((uint)(fromIdx));
    uint toIdxU32 = ((uint)(toIdx));
    uint fromBase = (fromIdxU32 * nComp);
    uint toBase = (toIdxU32 * nComp);
    float distSq = as_type<float>(0x38d1b717u);
    for (int d = 0; d < (int)(nComp); ++d) {
      uint dU32 = ((uint)(d));
      uint fromAddr = (fromBase + dU32);
      uint toAddr = (toBase + dU32);
      float fromVal = embedding[fromAddr];
      float toVal = embedding[toAddr];
      float diff = (fromVal - toVal);
      float diffSq = (diff * diff);
      float currDistSq = distSq;
      float newDistSq = (currDistSq + diffSq);
      distSq = newDistSq;
    }
    float dSq = distSq;
    float bMinus1 = (b - as_type<float>(0x3f800000u));
    float logDSq = log(dSq);
    float powTerm = exp((bMinus1 * logDSq));
    float dPow2b = exp((b * logDSq));
    float denom = (as_type<float>(0x3f800000u) + (a * dPow2b));
    float gradScale = ((((as_type<float>(0xc0000000u) * a) * b) * (weight * powTerm)) / denom);
    for (int d = 0; d < (int)(nComp); ++d) {
      uint dU32 = ((uint)(d));
      uint fromAddr = (fromBase + dU32);
      uint toAddr = (toBase + dU32);
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
