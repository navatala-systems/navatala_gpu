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
__kernel void navatala_ml_compute_perplexity_f64(__global const double* distances, __global const double* targetPerplexity, __global const uint* nPoints, __global const uint* maxIter, __global double* sigma) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  double perpTarget = targetPerplexity[0];
  uint maxIt = maxIter[0];
  double logPerpTarget = log(perpTarget);
  bool inBounds = (gid < n);
  if (inBounds) {
    uint i = gid;
    double sigmaLowAccum = as_double(0x3f1a36e2eb1c432dul);
    double sigmaHighAccum = as_double(0x40c3880000000000ul);
    double sigmaMidAccum = as_double(0x3ff0000000000000ul);
    for (int iter = 0; iter < (int)(maxIt); ++iter) {
      uint iterU32 = ((uint)(iter));
      double sigmaLow = sigmaLowAccum;
      double sigmaHigh = sigmaHighAccum;
      double sigmaMid = (as_double(0x3fe0000000000000ul) * (sigmaLow + sigmaHigh));
      sigmaMidAccum = sigmaMid;
      double twoSigmaSq = (as_double(0x4000000000000000ul) * (sigmaMid * sigmaMid));
      double sumPAccum = as_double(0x0000000000000000ul);
      double entropyAccum = as_double(0x0000000000000000ul);
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          uint distIdx = ((i * n) + jU32);
          double dij = distances[distIdx];
          double negDOverSigma = ((as_double(0x0000000000000000ul) - dij) / twoSigmaSq);
          double pVal = exp(negDOverSigma);
          double currentSumP = sumPAccum;
          double newSumP = (currentSumP + pVal);
          sumPAccum = newSumP;
        }
      }
      double sumP = sumPAccum;
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          uint distIdx = ((i * n) + jU32);
          double dij = distances[distIdx];
          double negDOverSigma = ((as_double(0x0000000000000000ul) - dij) / twoSigmaSq);
          double pVal = exp(negDOverSigma);
          double pNorm = (pVal / sumP);
          bool pGtZero = (pNorm > as_double(0x3e7ad7f29abcaf48ul));
          if (pGtZero) {
            double logP = log(pNorm);
            double entropyTerm = ((as_double(0x0000000000000000ul) - pNorm) * logP);
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
