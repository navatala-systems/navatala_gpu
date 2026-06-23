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
extern "C" __global__ void navatala_dataframe_explained_variance_f64(const double* y_true, const double* y_pred, const double* mean_residuals, const double* mean_true, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata_var_res[256];
  __shared__ double sdata_var_true[256];
  unsigned int countVal = n[0];
  double meanResVal = mean_residuals[0];
  double meanTrueVal = mean_true[0];
  bool inBounds = (gid < countVal);
  double trueVal = y_true[gid];
  double predVal = y_pred[gid];
  double residual = (trueVal - predVal);
  double resDiff = (residual - meanResVal);
  double sqResDiff = (resDiff * resDiff);
  double trueDiff = (trueVal - meanTrueVal);
  double sqTrueDiff = (trueDiff * trueDiff);
  if (inBounds) {
    sdata_var_res[lid] = sqResDiff;
    sdata_var_true[lid] = sqTrueDiff;
  } else {
    sdata_var_res[lid] = __longlong_as_double(0x0000000000000000ull);
    sdata_var_true[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int evF64_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int evF64_stride = evF64_reductionStride;
    if (lid < evF64_stride) {
      unsigned int evF64_partnerIdx = (lid + evF64_stride);
      double otherRes = sdata_var_res[evF64_partnerIdx];
      double mineRes = sdata_var_res[lid];
      double sumRes = (mineRes + otherRes);
      sdata_var_res[lid] = sumRes;
      double otherTrue = sdata_var_true[evF64_partnerIdx];
      double mineTrue = sdata_var_true[lid];
      double sumTrue = (mineTrue + otherTrue);
      sdata_var_true[lid] = sumTrue;
    }
    unsigned int evF64_strideToHalve = evF64_reductionStride;
    unsigned int evF64_nextStride = (evF64_strideToHalve >> 1u);
    evF64_reductionStride = evF64_nextStride;
    __syncthreads();
  }
  unsigned int zeroU32 = ((unsigned int)(0));
  if (lid == zeroU32) {
    double sumSqRes = sdata_var_res[0];
    double sumSqTrue = sdata_var_true[0];
    double countFloat = ((double)(countVal));
    double varRatio = (sumSqRes / sumSqTrue);
    double evVal = (__longlong_as_double(0x3ff0000000000000ull) - varRatio);
    result[0] = evVal;
  }
}
