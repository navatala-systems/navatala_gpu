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
extern "C" __global__ void navatala_ml_compute_repulsion_f32(const float* embedding, const int* negative_samples, const unsigned int* n_samples, const unsigned int* n_neg, const unsigned int* n_components, const float* a_param, const float* b_param, const float* gamma, float* gradients) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nNeg = n_neg[0];
  unsigned int nComp = n_components[0];
  float a = a_param[0];
  float b = b_param[0];
  float gammaVal = gamma[0];
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
        float distSqNeg = __uint_as_float(0x38d1b717u);
        for (int d = 0; d < (int)(nComp); ++d) {
          unsigned int dU32 = ((unsigned int)(d));
          unsigned int iAddr = (sampleBase + dU32);
          unsigned int jAddr = (negEmbBase + dU32);
          float iVal = embedding[iAddr];
          float jVal = embedding[jAddr];
          float diff = (iVal - jVal);
          float diffSq = (diff * diff);
          float currDSq = distSqNeg;
          float newDSq = (currDSq + diffSq);
          distSqNeg = newDSq;
        }
        float dSq = distSqNeg;
        float logDSqNeg = log(dSq);
        float dPow2b = exp((b * logDSqNeg));
        float denom = (__uint_as_float(0x3f800000u) + (a * dPow2b));
        float gradScale = (((__uint_as_float(0x40000000u) * gammaVal) * b) / (dSq * denom));
        for (int d = 0; d < (int)(nComp); ++d) {
          unsigned int dU32 = ((unsigned int)(d));
          unsigned int iAddr = (sampleBase + dU32);
          unsigned int jAddr = (negEmbBase + dU32);
          float iVal = embedding[iAddr];
          float jVal = embedding[jAddr];
          float diff = (iVal - jVal);
          float grad = (gradScale * diff);
          float gradClamped = (((grad > __uint_as_float(0x40800000u))) ? (__uint_as_float(0x40800000u)) : ((((grad < __uint_as_float(0xc0800000u))) ? (__uint_as_float(0xc0800000u)) : (grad))));
          float currGrad = gradients[iAddr];
          float newGrad = (currGrad + gradClamped);
          gradients[iAddr] = newGrad;
        }
      }
    }
  }
}
