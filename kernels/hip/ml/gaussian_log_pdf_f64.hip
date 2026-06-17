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
extern "C" __global__ void navatala_ml_gaussian_log_pdf_f64(const double* data, const double* means, const double* variances, const unsigned int* n, const unsigned int* k, const unsigned int* d, double* logPdf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numK = k[0];
  unsigned int numD = d[0];
  unsigned int nk = (numN * numK);
  bool inBounds = (gid < nk);
  if (inBounds) {
    unsigned int i = (gid / numK);
    unsigned int j = (gid % numK);
    double log2Pi = __longlong_as_double(0x3ffd67f1c864beb4ull);
    double dFloat = ((double)(numD));
    double dLog2Pi = (dFloat * log2Pi);
    double logVarSumAccum = __longlong_as_double(0x0000000000000000ull);
    double mahalSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int l = 0; l < (int)(numD); ++l) {
      unsigned int lU32 = ((unsigned int)(l));
      unsigned int dataIdx = ((i * numD) + lU32);
      unsigned int paramIdx = ((j * numD) + lU32);
      double x_il = data[dataIdx];
      double mu_jl = means[paramIdx];
      double var_jl = variances[paramIdx];
      double logVar = log(var_jl);
      double currLogVarSum = logVarSumAccum;
      double newLogVarSum = (currLogVarSum + logVar);
      logVarSumAccum = newLogVarSum;
      double diff = (x_il - mu_jl);
      double diffSq = (diff * diff);
      double mahal = (diffSq / var_jl);
      double currMahalSum = mahalSumAccum;
      double newMahalSum = (currMahalSum + mahal);
      mahalSumAccum = newMahalSum;
    }
    double logVarSum = logVarSumAccum;
    double mahalSum = mahalSumAccum;
    double innerSum = (dLog2Pi + (logVarSum + mahalSum));
    double logPdfVal = (__longlong_as_double(0xbfe0000000000000ull) * innerSum);
    logPdf[gid] = logPdfVal;
  }
}
