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

kernel void navatala_graph_rbf_affinity_f32(device const float* data [[buffer(0)]], device const float* sigma [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* d [[buffer(3)]], device float* W [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint dVal = d[0];
  float sigmaVal = sigma[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = as_type<float>(0x00000000u);
    } else {
      float distSq = as_type<float>(0x00000000u);
      uint rowI = (i * dVal);
      uint rowJ = (j * dVal);
      uint idxI = rowI;
      uint idxJ = rowJ;
      float xi = data[idxI];
      float xj = data[idxJ];
      float diff = (xi - xj);
      float diffSq = (diff * diff);
      float sigmaSq = (sigmaVal * sigmaVal);
      float twoSigmaSq = (as_type<float>(0x40000000u) * sigmaSq);
      float exponent = ((as_type<float>(0x00000000u) - diffSq) / twoSigmaSq);
      float affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}
