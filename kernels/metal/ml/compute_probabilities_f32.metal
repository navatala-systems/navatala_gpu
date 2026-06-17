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

kernel void navatala_ml_compute_probabilities_f32(device const int* labels [[buffer(0)]], device const float* lambdas [[buffer(1)]], device const float* cluster_lambdas_max [[buffer(2)]], device const uint* n_samples [[buffer(3)]], device const uint* n_clusters [[buffer(4)]], device float* probabilities [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int label = labels[gid];
    bool isNoise = (label < 0);
    if (isNoise) {
      probabilities[gid] = as_type<float>(0x00000000u);
    } else {
      float lambda = lambdas[gid];
      uint labelU32 = ((uint)(label));
      float lambdaMax = cluster_lambdas_max[labelU32];
      bool lambdaMaxIsZero = (lambdaMax <= as_type<float>(0x00000000u));
      float prob = as_type<float>(0x00000000u);
      if (lambdaMaxIsZero) {
        prob = as_type<float>(0x3f800000u);
      } else {
        float rawProb = (lambda / lambdaMax);
        bool probGt1 = (rawProb > as_type<float>(0x3f800000u));
        if (probGt1) {
          prob = as_type<float>(0x3f800000u);
        } else {
          bool probLt0 = (rawProb < as_type<float>(0x00000000u));
          if (probLt0) {
            prob = as_type<float>(0x00000000u);
          } else {
            prob = rawProb;
          }
        }
      }
      float finalProb = prob;
      probabilities[gid] = finalProb;
    }
  }
}
