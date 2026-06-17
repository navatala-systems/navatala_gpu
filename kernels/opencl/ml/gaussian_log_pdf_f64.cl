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
__kernel void navatala_ml_gaussian_log_pdf_f64(__global const double* data, __global const double* means, __global const double* variances, __global const uint* n, __global const uint* k, __global const uint* d, __global double* logPdf) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  uint numK = k[0];
  uint numD = d[0];
  uint nk = (numN * numK);
  bool inBounds = (gid < nk);
  if (inBounds) {
    uint i = (gid / numK);
    uint j = (gid % numK);
    double log2Pi = as_double(0x3ffd67f1c864beb4ul);
    double dFloat = ((double)(numD));
    double dLog2Pi = (dFloat * log2Pi);
    double logVarSumAccum = as_double(0x0000000000000000ul);
    double mahalSumAccum = as_double(0x0000000000000000ul);
    for (int l = 0; l < (int)(numD); ++l) {
      uint lU32 = ((uint)(l));
      uint dataIdx = ((i * numD) + lU32);
      uint paramIdx = ((j * numD) + lU32);
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
    double logPdfVal = (as_double(0xbfe0000000000000ul) * innerSum);
    logPdf[gid] = logPdfVal;
  }
}
