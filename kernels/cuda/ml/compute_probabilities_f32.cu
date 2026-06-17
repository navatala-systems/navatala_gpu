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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_compute_probabilities_f32(const int* labels, const float* lambdas, const float* cluster_lambdas_max, const unsigned int* n_samples, const unsigned int* n_clusters, float* probabilities) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int label = labels[gid];
    bool isNoise = (label < 0);
    if (isNoise) {
      probabilities[gid] = __uint_as_float(0x00000000u);
    } else {
      float lambda = lambdas[gid];
      unsigned int labelU32 = ((unsigned int)(label));
      float lambdaMax = cluster_lambdas_max[labelU32];
      bool lambdaMaxIsZero = (lambdaMax <= __uint_as_float(0x00000000u));
      float prob = __uint_as_float(0x00000000u);
      if (lambdaMaxIsZero) {
        prob = __uint_as_float(0x3f800000u);
      } else {
        float rawProb = (lambda / lambdaMax);
        bool probGt1 = (rawProb > __uint_as_float(0x3f800000u));
        if (probGt1) {
          prob = __uint_as_float(0x3f800000u);
        } else {
          bool probLt0 = (rawProb < __uint_as_float(0x00000000u));
          if (probLt0) {
            prob = __uint_as_float(0x00000000u);
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
