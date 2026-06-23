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
__kernel void navatala_ml_compute_log_likelihood_f64(__global const double* logPdf, __global const double* weights, __global const uint* n, __global const uint* k, __global double* logLikelihood) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if (gid == (uint)(0u)) {
    uint numN = n[0];
    uint numK = k[0];
    double totalLLAccum = as_double(0x0000000000000000ul);
    double maxLogProbAccum = as_double(0xffe1ccf385ebc8a0ul);
    double expSumAccum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      maxLogProbAccum = as_double(0xffe1ccf385ebc8a0ul);
      for (int j = 0; j < (int)(numK); ++j) {
        uint jU32 = ((uint)(j));
        uint logPdfIdx = ((iU32 * numK) + jU32);
        double logP = logPdf[logPdfIdx];
        double w_j = weights[jU32];
        double logW = log(w_j);
        double logProb = (logW + logP);
        double currMax = maxLogProbAccum;
        bool isGreater = (logProb > currMax);
        double newMax = ((isGreater) ? (logProb) : (currMax));
        maxLogProbAccum = newMax;
      }
      double maxLogProb = maxLogProbAccum;
      expSumAccum = as_double(0x0000000000000000ul);
      for (int j = 0; j < (int)(numK); ++j) {
        uint jU32 = ((uint)(j));
        uint logPdfIdx = ((iU32 * numK) + jU32);
        double logP = logPdf[logPdfIdx];
        double w_j = weights[jU32];
        double logW = log(w_j);
        double logProb = (logW + logP);
        double shifted = (logProb - maxLogProb);
        double expVal = exp(shifted);
        double currExpSum = expSumAccum;
        double newExpSum = (currExpSum + expVal);
        expSumAccum = newExpSum;
      }
      double expSum = expSumAccum;
      double logExpSum = log(expSum);
      double logSumExp = (maxLogProb + logExpSum);
      double currTotalLL = totalLLAccum;
      double newTotalLL = (currTotalLL + logSumExp);
      totalLLAccum = newTotalLL;
    }
    double totalLL = totalLLAccum;
    logLikelihood[0] = totalLL;
  }
}
