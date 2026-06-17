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
extern "C" __global__ void navatala_ml_compute_log_likelihood_f32(const float* logPdf, const float* weights, const unsigned int* n, const unsigned int* k, float* logLikelihood) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gid == 0u)) {
    unsigned int numN = n[0];
    unsigned int numK = k[0];
    float totalLLAccum = __uint_as_float(0x00000000u);
    float maxLogProbAccum = __uint_as_float(0xfe967699u);
    float expSumAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      maxLogProbAccum = __uint_as_float(0xfe967699u);
      for (int j = 0; j < (int)(numK); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        unsigned int logPdfIdx = ((iU32 * numK) + jU32);
        float logP = logPdf[logPdfIdx];
        float w_j = weights[jU32];
        float logW = log(w_j);
        float logProb = (logW + logP);
        float currMax = maxLogProbAccum;
        bool isGreater = (logProb > currMax);
        float newMax = ((isGreater) ? (logProb) : (currMax));
        maxLogProbAccum = newMax;
      }
      float maxLogProb = maxLogProbAccum;
      expSumAccum = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(numK); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        unsigned int logPdfIdx = ((iU32 * numK) + jU32);
        float logP = logPdf[logPdfIdx];
        float w_j = weights[jU32];
        float logW = log(w_j);
        float logProb = (logW + logP);
        float shifted = (logProb - maxLogProb);
        float expVal = exp(shifted);
        float currExpSum = expSumAccum;
        float newExpSum = (currExpSum + expVal);
        expSumAccum = newExpSum;
      }
      float expSum = expSumAccum;
      float logExpSum = log(expSum);
      float logSumExp = (maxLogProb + logExpSum);
      float currTotalLL = totalLLAccum;
      float newTotalLL = (currTotalLL + logSumExp);
      totalLLAccum = newTotalLL;
    }
    float totalLL = totalLLAccum;
    logLikelihood[0] = totalLL;
  }
}
