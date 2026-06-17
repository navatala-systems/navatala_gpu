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

__kernel void navatala_ml_svm_predict_classify_f32(__global const float* support_vectors, __global const float* dual_coefs, __global const float* X, __global const float* bias, __global const float* gamma, __global const uint* n_samples, __global const uint* n_sv, __global const uint* n_features, __global int* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nSV = n_sv[0];
  uint nFeatures = n_features[0];
  float biasVal = bias[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float decisionSum = as_float(0x00000000u);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      uint svU32 = ((uint)(sv));
      uint svBase = (svU32 * nFeatures);
      float sqDist = as_float(0x00000000u);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        uint fU32 = ((uint)(f));
        uint xIdx = (sampleBase + fU32);
        uint svIdx = (svBase + fU32);
        float xVal = X[xIdx];
        float svVal = support_vectors[svIdx];
        float diff = (xVal - svVal);
        float diffSq = (diff * diff);
        float currSqDist = sqDist;
        float newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      float finalSqDist = sqDist;
      float negGammaDist = (as_float(0xbf800000u) * (gammaVal * finalSqDist));
      float kernelVal = exp(negGammaDist);
      float alpha = dual_coefs[svU32];
      float contribution = (alpha * kernelVal);
      float currSum = decisionSum;
      float newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    float finalSum = decisionSum;
    float decision = (finalSum + biasVal);
    bool isPositive = (decision >= as_float(0x00000000u));
    if (isPositive) {
      predictions[gid] = 1;
    } else {
      predictions[gid] = -1;
    }
  }
}
