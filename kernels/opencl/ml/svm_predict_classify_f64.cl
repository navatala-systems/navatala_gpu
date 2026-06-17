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
__kernel void navatala_ml_svm_predict_classify_f64(__global const double* support_vectors, __global const double* dual_coefs, __global const double* X, __global const double* bias, __global const double* gamma, __global const uint* n_samples, __global const uint* n_sv, __global const uint* n_features, __global int* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nSV = n_sv[0];
  uint nFeatures = n_features[0];
  double biasVal = bias[0];
  double gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    double decisionSum = as_double(0x0000000000000000ul);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      uint svU32 = ((uint)(sv));
      uint svBase = (svU32 * nFeatures);
      double sqDist = as_double(0x0000000000000000ul);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        uint fU32 = ((uint)(f));
        uint xIdx = (sampleBase + fU32);
        uint svIdx = (svBase + fU32);
        double xVal = X[xIdx];
        double svVal = support_vectors[svIdx];
        double diff = (xVal - svVal);
        double diffSq = (diff * diff);
        double currSqDist = sqDist;
        double newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      double finalSqDist = sqDist;
      double negGammaDist = (as_double(0xbff0000000000000ul) * (gammaVal * finalSqDist));
      double kernelVal = exp(negGammaDist);
      double alpha = dual_coefs[svU32];
      double contribution = (alpha * kernelVal);
      double currSum = decisionSum;
      double newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    double finalSum = decisionSum;
    double decision = (finalSum + biasVal);
    bool isPositive = (decision >= as_double(0x0000000000000000ul));
    if (isPositive) {
      predictions[gid] = 1;
    } else {
      predictions[gid] = -1;
    }
  }
}
