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
extern "C" __global__ void navatala_ml_gaussian_log_pdf_f32(const float* data, const float* means, const float* variances, const unsigned int* n, const unsigned int* k, const unsigned int* d, float* logPdf) {
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
    float log2Pi = __uint_as_float(0x3feb3f8eu);
    float dFloat = ((float)(numD));
    float dLog2Pi = (dFloat * log2Pi);
    float logVarSumAccum = __uint_as_float(0x00000000u);
    float mahalSumAccum = __uint_as_float(0x00000000u);
    for (int l = 0; l < (int)(numD); ++l) {
      unsigned int lU32 = ((unsigned int)(l));
      unsigned int dataIdx = ((i * numD) + lU32);
      unsigned int paramIdx = ((j * numD) + lU32);
      float x_il = data[dataIdx];
      float mu_jl = means[paramIdx];
      float var_jl = variances[paramIdx];
      float logVar = log(var_jl);
      float currLogVarSum = logVarSumAccum;
      float newLogVarSum = (currLogVarSum + logVar);
      logVarSumAccum = newLogVarSum;
      float diff = (x_il - mu_jl);
      float diffSq = (diff * diff);
      float mahal = (diffSq / var_jl);
      float currMahalSum = mahalSumAccum;
      float newMahalSum = (currMahalSum + mahal);
      mahalSumAccum = newMahalSum;
    }
    float logVarSum = logVarSumAccum;
    float mahalSum = mahalSumAccum;
    float innerSum = (dLog2Pi + (logVarSum + mahalSum));
    float logPdfVal = (__uint_as_float(0xbf000000u) * innerSum);
    logPdf[gid] = logPdfVal;
  }
}
