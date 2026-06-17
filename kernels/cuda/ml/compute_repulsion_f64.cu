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
extern "C" __global__ void navatala_ml_compute_repulsion_f64(const double* embedding, const int* negative_samples, const unsigned int* n_samples, const unsigned int* n_neg, const unsigned int* n_components, const double* a_param, const double* b_param, const double* gamma, double* gradients) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nNeg = n_neg[0];
  unsigned int nComp = n_components[0];
  double a = a_param[0];
  double b = b_param[0];
  double gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nComp);
    unsigned int negBase = (gid * nNeg);
    for (int n = 0; n < (int)(nNeg); ++n) {
      unsigned int nU32 = ((unsigned int)(n));
      unsigned int negSampleIdx = (negBase + nU32);
      int negIdx = negative_samples[negSampleIdx];
      bool validNeg = (negIdx >= 0);
      if (validNeg) {
        unsigned int negIdxU32 = ((unsigned int)(negIdx));
        unsigned int negEmbBase = (negIdxU32 * nComp);
        double distSqNeg = __longlong_as_double(0x3f1a36e2eb1c432dull);
        for (int d = 0; d < (int)(nComp); ++d) {
          unsigned int dU32 = ((unsigned int)(d));
          unsigned int iAddr = (sampleBase + dU32);
          unsigned int jAddr = (negEmbBase + dU32);
          double iVal = embedding[iAddr];
          double jVal = embedding[jAddr];
          double diff = (iVal - jVal);
          double diffSq = (diff * diff);
          double currDSq = distSqNeg;
          double newDSq = (currDSq + diffSq);
          distSqNeg = newDSq;
        }
        double dSq = distSqNeg;
        double logDSqNeg = log(dSq);
        double dPow2b = exp((b * logDSqNeg));
        double denom = (__longlong_as_double(0x3ff0000000000000ull) + (a * dPow2b));
        double gradScale = (((__longlong_as_double(0x4000000000000000ull) * gammaVal) * b) / (dSq * denom));
        for (int d = 0; d < (int)(nComp); ++d) {
          unsigned int dU32 = ((unsigned int)(d));
          unsigned int iAddr = (sampleBase + dU32);
          unsigned int jAddr = (negEmbBase + dU32);
          double iVal = embedding[iAddr];
          double jVal = embedding[jAddr];
          double diff = (iVal - jVal);
          double grad = (gradScale * diff);
          double gradClamped = (((grad > __longlong_as_double(0x4010000000000000ull))) ? (__longlong_as_double(0x4010000000000000ull)) : ((((grad < __longlong_as_double(0xc010000000000000ull))) ? (__longlong_as_double(0xc010000000000000ull)) : (grad))));
          double currGrad = gradients[iAddr];
          double newGrad = (currGrad + gradClamped);
          gradients[iAddr] = newGrad;
        }
      }
    }
  }
}
