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
extern "C" __global__ void navatala_ml_compute_decision_function_f32(const float* support_vectors, const float* dual_coefs, const float* X, const float* bias, const float* gamma, const unsigned int* n_samples, const unsigned int* n_sv, const unsigned int* n_features, float* decision_values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nSV = n_sv[0];
  unsigned int nFeatures = n_features[0];
  float biasVal = bias[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nFeatures);
    float decisionSum = __uint_as_float(0x00000000u);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      unsigned int svU32 = ((unsigned int)(sv));
      unsigned int svBase = (svU32 * nFeatures);
      float sqDist = __uint_as_float(0x00000000u);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        unsigned int fU32 = ((unsigned int)(f));
        unsigned int xIdx = (sampleBase + fU32);
        unsigned int svIdx = (svBase + fU32);
        float xVal = X[xIdx];
        float svVal = support_vectors[svIdx];
        float diff = (xVal - svVal);
        float diffSq = (diff * diff);
        float currSqDist = sqDist;
        float newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      float finalSqDist = sqDist;
      float negGammaDist = (__uint_as_float(0xbf800000u) * (gammaVal * finalSqDist));
      float kernelVal = exp(negGammaDist);
      float alpha = dual_coefs[svU32];
      float contribution = (alpha * kernelVal);
      float currSum = decisionSum;
      float newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    float finalSum = decisionSum;
    float decision = (finalSum + biasVal);
    decision_values[gid] = decision;
  }
}
