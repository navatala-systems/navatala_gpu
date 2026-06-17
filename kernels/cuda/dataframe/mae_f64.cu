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
extern "C" __global__ void navatala_dataframe_mae_f64(const double* y_true, const double* y_pred, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int countVal = n[0];
  bool inBounds = (gid < countVal);
  double trueVal = y_true[gid];
  double predVal = y_pred[gid];
  double diff = (trueVal - predVal);
  double absErr = abs(diff);
  if (inBounds) {
    sdata[lid] = absErr;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int maeF64_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int maeF64_stride = maeF64_reductionStride;
    if ((lid < maeF64_stride)) {
      unsigned int maeF64_partnerIdx = (lid + maeF64_stride);
      double other = sdata[maeF64_partnerIdx];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int maeF64_strideToHalve = maeF64_reductionStride;
    unsigned int maeF64_nextStride = (maeF64_strideToHalve >> 1u);
    maeF64_reductionStride = maeF64_nextStride;
    __syncthreads();
  }
  unsigned int zeroU32 = ((unsigned int)(0));
  if ((lid == zeroU32)) {
    double totalSum = sdata[0];
    double countFloat = ((double)(countVal));
    double maeVal = (totalSum / countFloat);
    result[0] = maeVal;
  }
}
