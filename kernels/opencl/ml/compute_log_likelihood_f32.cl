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

__kernel void navatala_ml_compute_log_likelihood_f32(__global const float* logPdf, __global const float* weights, __global const uint* n, __global const uint* k, __global float* logLikelihood) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    uint numN = n[0];
    uint numK = k[0];
    float totalLLAccum = as_float(0x00000000u);
    float maxLogProbAccum = as_float(0xfe967699u);
    float expSumAccum = as_float(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      maxLogProbAccum = as_float(0xfe967699u);
      for (int j = 0; j < (int)(numK); ++j) {
        uint jU32 = ((uint)(j));
        uint logPdfIdx = ((iU32 * numK) + jU32);
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
      expSumAccum = as_float(0x00000000u);
      for (int j = 0; j < (int)(numK); ++j) {
        uint jU32 = ((uint)(j));
        uint logPdfIdx = ((iU32 * numK) + jU32);
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
