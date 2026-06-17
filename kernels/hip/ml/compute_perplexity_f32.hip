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
extern "C" __global__ void navatala_ml_compute_perplexity_f32(const float* distances, const float* targetPerplexity, const unsigned int* nPoints, const unsigned int* maxIter, float* sigma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  float perpTarget = targetPerplexity[0];
  unsigned int maxIt = maxIter[0];
  float logPerpTarget = log(perpTarget);
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int i = gid;
    float sigmaLowAccum = __uint_as_float(0x38d1b717u);
    float sigmaHighAccum = __uint_as_float(0x461c4000u);
    float sigmaMidAccum = __uint_as_float(0x3f800000u);
    for (int iter = 0; iter < (int)(maxIt); ++iter) {
      unsigned int iterU32 = ((unsigned int)(iter));
      float sigmaLow = sigmaLowAccum;
      float sigmaHigh = sigmaHighAccum;
      float sigmaMid = (__uint_as_float(0x3f000000u) * (sigmaLow + sigmaHigh));
      sigmaMidAccum = sigmaMid;
      float twoSigmaSq = (__uint_as_float(0x40000000u) * (sigmaMid * sigmaMid));
      float sumPAccum = __uint_as_float(0x00000000u);
      float entropyAccum = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(n); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          unsigned int distIdx = ((i * n) + jU32);
          float dij = distances[distIdx];
          float negDOverSigma = ((__uint_as_float(0x00000000u) - dij) / twoSigmaSq);
          float pVal = exp(negDOverSigma);
          float currentSumP = sumPAccum;
          float newSumP = (currentSumP + pVal);
          sumPAccum = newSumP;
        }
      }
      float sumP = sumPAccum;
      for (int j = 0; j < (int)(n); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          unsigned int distIdx = ((i * n) + jU32);
          float dij = distances[distIdx];
          float negDOverSigma = ((__uint_as_float(0x00000000u) - dij) / twoSigmaSq);
          float pVal = exp(negDOverSigma);
          float pNorm = (pVal / sumP);
          bool pGtZero = (pNorm > __uint_as_float(0x33d6bf95u));
          if (pGtZero) {
            float logP = log(pNorm);
            float entropyTerm = ((__uint_as_float(0x00000000u) - pNorm) * logP);
            float currentEntropy = entropyAccum;
            float newEntropy = (currentEntropy + entropyTerm);
            entropyAccum = newEntropy;
          }
        }
      }
      float entropy = entropyAccum;
      bool entropyTooHigh = (entropy > logPerpTarget);
      if (entropyTooHigh) {
        sigmaHighAccum = sigmaMid;
      } else {
        sigmaLowAccum = sigmaMid;
      }
    }
    float finalSigma = sigmaMidAccum;
    sigma[gid] = finalSigma;
  }
}
