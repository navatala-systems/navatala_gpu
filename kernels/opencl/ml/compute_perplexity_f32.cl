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

__kernel void navatala_ml_compute_perplexity_f32(__global const float* distances, __global const float* targetPerplexity, __global const uint* nPoints, __global const uint* maxIter, __global float* sigma) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  float perpTarget = targetPerplexity[0];
  uint maxIt = maxIter[0];
  float logPerpTarget = log(perpTarget);
  bool inBounds = (gid < n);
  if (inBounds) {
    uint i = gid;
    float sigmaLowAccum = as_float(0x38d1b717u);
    float sigmaHighAccum = as_float(0x461c4000u);
    float sigmaMidAccum = as_float(0x3f800000u);
    for (int iter = 0; iter < (int)(maxIt); ++iter) {
      uint iterU32 = ((uint)(iter));
      float sigmaLow = sigmaLowAccum;
      float sigmaHigh = sigmaHighAccum;
      float sigmaMid = (as_float(0x3f000000u) * (sigmaLow + sigmaHigh));
      sigmaMidAccum = sigmaMid;
      float twoSigmaSq = (as_float(0x40000000u) * (sigmaMid * sigmaMid));
      float sumPAccum = as_float(0x00000000u);
      float entropyAccum = as_float(0x00000000u);
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          uint distIdx = ((i * n) + jU32);
          float dij = distances[distIdx];
          float negDOverSigma = ((as_float(0x00000000u) - dij) / twoSigmaSq);
          float pVal = exp(negDOverSigma);
          float currentSumP = sumPAccum;
          float newSumP = (currentSumP + pVal);
          sumPAccum = newSumP;
        }
      }
      float sumP = sumPAccum;
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          uint distIdx = ((i * n) + jU32);
          float dij = distances[distIdx];
          float negDOverSigma = ((as_float(0x00000000u) - dij) / twoSigmaSq);
          float pVal = exp(negDOverSigma);
          float pNorm = (pVal / sumP);
          bool pGtZero = (pNorm > as_float(0x33d6bf95u));
          if (pGtZero) {
            float logP = log(pNorm);
            float entropyTerm = ((as_float(0x00000000u) - pNorm) * logP);
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
