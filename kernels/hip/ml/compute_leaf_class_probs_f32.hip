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
extern "C" __global__ void navatala_ml_compute_leaf_class_probs_f32(const int* labels, const unsigned int* sample_indices, const unsigned int* n_leaf_samples, const unsigned int* n_classes, float* probs, unsigned int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_leaf_samples[0];
  unsigned int nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    unsigned int labelU32 = ((unsigned int)(label));
    unsigned int currCount = counts[labelU32];
    unsigned int newCount = (currCount + 1u);
    counts[labelU32] = newCount;
  }
  __syncthreads();
  bool inBoundsClass = (gid < nClasses);
  if (inBoundsClass) {
    unsigned int count = counts[gid];
    float countF = ((float)(count));
    float nSamplesF = ((float)(nSamples));
    float prob = (countF / nSamplesF);
    probs[gid] = prob;
  }
}
