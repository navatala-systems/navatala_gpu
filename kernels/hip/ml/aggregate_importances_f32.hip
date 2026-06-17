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
extern "C" __global__ void navatala_ml_aggregate_importances_f32(const float* tree_importances, const unsigned int* n_trees, const unsigned int* n_features, float* importances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nTrees = n_trees[0];
  unsigned int nFeatures = n_features[0];
  bool inBounds = (gid < nFeatures);
  if (inBounds) {
    float sumImportance = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      unsigned int tU32 = ((unsigned int)(t));
      unsigned int idx = ((tU32 * nFeatures) + gid);
      float imp = tree_importances[idx];
      float currSum = sumImportance;
      float newSum = (currSum + imp);
      sumImportance = newSum;
    }
    float finalSum = sumImportance;
    float nTreesF = ((float)(nTrees));
    float avgImp = (finalSum / nTreesF);
    importances[gid] = avgImp;
  }
}
