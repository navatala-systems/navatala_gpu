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
extern "C" __global__ void navatala_ml_compute_gini_impurity_f64(const int* labels, const unsigned int* sample_indices, const unsigned int* n_node_samples, const unsigned int* n_classes, double* gini, unsigned int* class_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_node_samples[0];
  unsigned int nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    unsigned int labelU32 = ((unsigned int)(label));
    unsigned int currCount = class_counts[labelU32];
    unsigned int newCount = (currCount + 1u);
    class_counts[labelU32] = newCount;
  }
  __syncthreads();
  bool isFirstThread = (gid == 0u);
  if (isFirstThread) {
    double sumPSquared = __longlong_as_double(0x0000000000000000ull);
    double nSamplesF = ((double)(nSamples));
    for (int c = 0; c < (int)(nClasses); ++c) {
      unsigned int cU32 = ((unsigned int)(c));
      unsigned int count = class_counts[cU32];
      double countF = ((double)(count));
      double p = (countF / nSamplesF);
      double pSquared = (p * p);
      double currSum = sumPSquared;
      double newSum = (currSum + pSquared);
      sumPSquared = newSum;
    }
    double finalSum = sumPSquared;
    double giniVal = (__longlong_as_double(0x3ff0000000000000ull) - finalSum);
    gini[0] = giniVal;
  }
}
