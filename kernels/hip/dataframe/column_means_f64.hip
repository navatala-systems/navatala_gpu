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
extern "C" __global__ void navatala_dataframe_column_means_f64(const double* _input, const unsigned int* nSamples, const unsigned int* nFeatures, double* means) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int colIdx = wgid;
  bool colInBounds = (colIdx < d);
  if (colInBounds) {
    double partialSum = __longlong_as_double(0x0000000000000000ull);
    unsigned int rowIdx = lid;
    bool rowInBounds = (rowIdx < n);
    unsigned int elemIdx = ((rowIdx * d) + colIdx);
    if (rowInBounds) {
      double val = _input[elemIdx];
      sdata[lid] = val;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int colMeansF64_reductionStride = 128u;
    for (int colMeansF64_reductionStep = 0; colMeansF64_reductionStep < (int)(8); ++colMeansF64_reductionStep) {
      unsigned int colMeansF64_stride = colMeansF64_reductionStride;
      if ((lid < colMeansF64_stride)) {
        double colMeansF64_other = sdata[(lid + colMeansF64_stride)];
        double colMeansF64_mine = sdata[lid];
        double colMeansF64_sum = (colMeansF64_mine + colMeansF64_other);
        sdata[lid] = colMeansF64_sum;
      }
      unsigned int colMeansF64_strideToHalve = colMeansF64_reductionStride;
      unsigned int colMeansF64_nextStride = (colMeansF64_strideToHalve >> 1u);
      colMeansF64_reductionStride = colMeansF64_nextStride;
      __syncthreads();
    }
    if ((lid == 0u)) {
      double totalSum = sdata[0u];
      double nFloat = ((double)(n));
      double meanVal = (totalSum / nFloat);
      means[colIdx] = meanVal;
    }
  }
}
