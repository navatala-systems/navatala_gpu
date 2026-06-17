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
extern "C" __global__ void navatala_ml_compute_perplexity_f64(const double* distances, const double* targetPerplexity, const unsigned int* nPoints, const unsigned int* maxIter, double* sigma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  double perpTarget = targetPerplexity[0];
  unsigned int maxIt = maxIter[0];
  double logPerpTarget = log(perpTarget);
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int i = gid;
    double sigmaLowAccum = __longlong_as_double(0x3f1a36e2eb1c432dull);
    double sigmaHighAccum = __longlong_as_double(0x40c3880000000000ull);
    double sigmaMidAccum = __longlong_as_double(0x3ff0000000000000ull);
    for (int iter = 0; iter < (int)(maxIt); ++iter) {
      unsigned int iterU32 = ((unsigned int)(iter));
      double sigmaLow = sigmaLowAccum;
      double sigmaHigh = sigmaHighAccum;
      double sigmaMid = (__longlong_as_double(0x3fe0000000000000ull) * (sigmaLow + sigmaHigh));
      sigmaMidAccum = sigmaMid;
      double twoSigmaSq = (__longlong_as_double(0x4000000000000000ull) * (sigmaMid * sigmaMid));
      double sumPAccum = __longlong_as_double(0x0000000000000000ull);
      double entropyAccum = __longlong_as_double(0x0000000000000000ull);
      for (int j = 0; j < (int)(n); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          unsigned int distIdx = ((i * n) + jU32);
          double dij = distances[distIdx];
          double negDOverSigma = ((__longlong_as_double(0x0000000000000000ull) - dij) / twoSigmaSq);
          double pVal = exp(negDOverSigma);
          double currentSumP = sumPAccum;
          double newSumP = (currentSumP + pVal);
          sumPAccum = newSumP;
        }
      }
      double sumP = sumPAccum;
      for (int j = 0; j < (int)(n); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          unsigned int distIdx = ((i * n) + jU32);
          double dij = distances[distIdx];
          double negDOverSigma = ((__longlong_as_double(0x0000000000000000ull) - dij) / twoSigmaSq);
          double pVal = exp(negDOverSigma);
          double pNorm = (pVal / sumP);
          bool pGtZero = (pNorm > __longlong_as_double(0x3e7ad7f29abcaf48ull));
          if (pGtZero) {
            double logP = log(pNorm);
            double entropyTerm = ((__longlong_as_double(0x0000000000000000ull) - pNorm) * logP);
            double currentEntropy = entropyAccum;
            double newEntropy = (currentEntropy + entropyTerm);
            entropyAccum = newEntropy;
          }
        }
      }
      double entropy = entropyAccum;
      bool entropyTooHigh = (entropy > logPerpTarget);
      if (entropyTooHigh) {
        sigmaHighAccum = sigmaMid;
      } else {
        sigmaLowAccum = sigmaMid;
      }
    }
    double finalSigma = sigmaMidAccum;
    sigma[gid] = finalSigma;
  }
}
