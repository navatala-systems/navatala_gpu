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
extern "C" __global__ void navatala_ml_svm_predict_classify_f64(const double* support_vectors, const double* dual_coefs, const double* X, const double* bias, const double* gamma, const unsigned int* n_samples, const unsigned int* n_sv, const unsigned int* n_features, int* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nSV = n_sv[0];
  unsigned int nFeatures = n_features[0];
  double biasVal = bias[0];
  double gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nFeatures);
    double decisionSum = __longlong_as_double(0x0000000000000000ull);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      unsigned int svU32 = ((unsigned int)(sv));
      unsigned int svBase = (svU32 * nFeatures);
      double sqDist = __longlong_as_double(0x0000000000000000ull);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        unsigned int fU32 = ((unsigned int)(f));
        unsigned int xIdx = (sampleBase + fU32);
        unsigned int svIdx = (svBase + fU32);
        double xVal = X[xIdx];
        double svVal = support_vectors[svIdx];
        double diff = (xVal - svVal);
        double diffSq = (diff * diff);
        double currSqDist = sqDist;
        double newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      double finalSqDist = sqDist;
      double negGammaDist = (__longlong_as_double(0xbff0000000000000ull) * (gammaVal * finalSqDist));
      double kernelVal = exp(negGammaDist);
      double alpha = dual_coefs[svU32];
      double contribution = (alpha * kernelVal);
      double currSum = decisionSum;
      double newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    double finalSum = decisionSum;
    double decision = (finalSum + biasVal);
    bool isPositive = (decision >= __longlong_as_double(0x0000000000000000ull));
    if (isPositive) {
      predictions[gid] = 1;
    } else {
      predictions[gid] = -1;
    }
  }
}
