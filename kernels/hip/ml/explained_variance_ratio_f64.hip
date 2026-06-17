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
extern "C" __global__ void navatala_ml_explained_variance_ratio_f64(const double* eigenvalues, const unsigned int* nFeatures, const unsigned int* nComponents, double* ratios) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int d = nFeatures[0];
  unsigned int kComp = nComponents[0];
  bool inBoundsLoad = (gid < d);
  if (inBoundsLoad) {
    double eigVal = eigenvalues[gid];
    sdata[lid] = eigVal;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int varRatioF64RedStride = 128u;
  for (int varRatioF64RedStep = 0; varRatioF64RedStep < (int)(8); ++varRatioF64RedStep) {
    unsigned int varRatioF64Stride = varRatioF64RedStride;
    if ((lid < varRatioF64Stride)) {
      double other = sdata[(lid + varRatioF64Stride)];
      double mine = sdata[lid];
      double sumVal = (mine + other);
      sdata[lid] = sumVal;
    }
    unsigned int varRatioF64StrideToHalve = varRatioF64RedStride;
    unsigned int varRatioF64NextStride = (varRatioF64StrideToHalve >> 1u);
    varRatioF64RedStride = varRatioF64NextStride;
    __syncthreads();
  }
  double totalVariance = sdata[0];
  __syncthreads();
  bool inBoundsRatio = (gid < kComp);
  if (inBoundsRatio) {
    double eigVal = eigenvalues[gid];
    double ratio = (eigVal / totalVariance);
    ratios[gid] = ratio;
  }
}
