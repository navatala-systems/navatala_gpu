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
extern "C" __global__ void navatala_ml_explained_variance_ratio_f32(const float* eigenvalues, const unsigned int* nFeatures, const unsigned int* nComponents, float* ratios) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int d = nFeatures[0];
  unsigned int kComp = nComponents[0];
  bool inBoundsLoad = (gid < d);
  if (inBoundsLoad) {
    float eigVal = eigenvalues[gid];
    sdata[lid] = eigVal;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int varRatioF32RedStride = 128u;
  for (int varRatioF32RedStep = 0; varRatioF32RedStep < (int)(8); ++varRatioF32RedStep) {
    unsigned int varRatioF32Stride = varRatioF32RedStride;
    if (lid < varRatioF32Stride) {
      float other = sdata[(lid + varRatioF32Stride)];
      float mine = sdata[lid];
      float sumVal = (mine + other);
      sdata[lid] = sumVal;
    }
    unsigned int varRatioF32StrideToHalve = varRatioF32RedStride;
    unsigned int varRatioF32NextStride = (varRatioF32StrideToHalve >> 1u);
    varRatioF32RedStride = varRatioF32NextStride;
    __syncthreads();
  }
  float totalVariance = sdata[0];
  __syncthreads();
  bool inBoundsRatio = (gid < kComp);
  if (inBoundsRatio) {
    float eigVal = eigenvalues[gid];
    float ratio = (eigVal / totalVariance);
    ratios[gid] = ratio;
  }
}
