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
extern "C" __global__ void navatala_dataframe_column_means_f32(const float* _input, const unsigned int* nSamples, const unsigned int* nFeatures, float* means) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ float sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int colIdx = wgid;
  bool colInBounds = (colIdx < d);
  if (colInBounds) {
    float partialSum = __uint_as_float(0x00000000u);
    unsigned int rowIdx = lid;
    bool rowInBounds = (rowIdx < n);
    unsigned int elemIdx = ((rowIdx * d) + colIdx);
    if (rowInBounds) {
      float val = _input[elemIdx];
      sdata[lid] = val;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int colMeansF32_reductionStride = 128u;
    for (int colMeansF32_reductionStep = 0; colMeansF32_reductionStep < (int)(8); ++colMeansF32_reductionStep) {
      unsigned int colMeansF32_stride = colMeansF32_reductionStride;
      if ((lid < colMeansF32_stride)) {
        float colMeansF32_other = sdata[(lid + colMeansF32_stride)];
        float colMeansF32_mine = sdata[lid];
        float colMeansF32_sum = (colMeansF32_mine + colMeansF32_other);
        sdata[lid] = colMeansF32_sum;
      }
      unsigned int colMeansF32_strideToHalve = colMeansF32_reductionStride;
      unsigned int colMeansF32_nextStride = (colMeansF32_strideToHalve >> 1u);
      colMeansF32_reductionStride = colMeansF32_nextStride;
      __syncthreads();
    }
    if ((lid == 0u)) {
      float totalSum = sdata[0u];
      float nFloat = ((float)(n));
      float meanVal = (totalSum / nFloat);
      means[colIdx] = meanVal;
    }
  }
}
