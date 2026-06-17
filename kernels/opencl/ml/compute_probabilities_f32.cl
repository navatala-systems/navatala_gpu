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

__kernel void navatala_ml_compute_probabilities_f32(__global const int* labels, __global const float* lambdas, __global const float* cluster_lambdas_max, __global const uint* n_samples, __global const uint* n_clusters, __global float* probabilities) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int label = labels[gid];
    bool isNoise = (label < 0);
    if (isNoise) {
      probabilities[gid] = as_float(0x00000000u);
    } else {
      float lambda = lambdas[gid];
      uint labelU32 = ((uint)(label));
      float lambdaMax = cluster_lambdas_max[labelU32];
      bool lambdaMaxIsZero = (lambdaMax <= as_float(0x00000000u));
      float prob = as_float(0x00000000u);
      if (lambdaMaxIsZero) {
        prob = as_float(0x3f800000u);
      } else {
        float rawProb = (lambda / lambdaMax);
        bool probGt1 = (rawProb > as_float(0x3f800000u));
        if (probGt1) {
          prob = as_float(0x3f800000u);
        } else {
          bool probLt0 = (rawProb < as_float(0x00000000u));
          if (probLt0) {
            prob = as_float(0x00000000u);
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
