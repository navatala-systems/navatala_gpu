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
__kernel void navatala_ml_compute_probabilities_f64(__global const int* labels, __global const double* lambdas, __global const double* cluster_lambdas_max, __global const uint* n_samples, __global const uint* n_clusters, __global double* probabilities) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int label = labels[gid];
    bool isNoise = (label < 0);
    if (isNoise) {
      probabilities[gid] = as_double(0x0000000000000000ul);
    } else {
      double lambda = lambdas[gid];
      uint labelU32 = ((uint)(label));
      double lambdaMax = cluster_lambdas_max[labelU32];
      bool lambdaMaxIsZero = (lambdaMax <= as_double(0x0000000000000000ul));
      double prob = as_double(0x0000000000000000ul);
      if (lambdaMaxIsZero) {
        prob = as_double(0x3ff0000000000000ul);
      } else {
        double rawProb = (lambda / lambdaMax);
        bool probGt1 = (rawProb > as_double(0x3ff0000000000000ul));
        if (probGt1) {
          prob = as_double(0x3ff0000000000000ul);
        } else {
          bool probLt0 = (rawProb < as_double(0x0000000000000000ul));
          if (probLt0) {
            prob = as_double(0x0000000000000000ul);
          } else {
            prob = rawProb;
          }
        }
      }
      double finalProb = prob;
      probabilities[gid] = finalProb;
    }
  }
}
