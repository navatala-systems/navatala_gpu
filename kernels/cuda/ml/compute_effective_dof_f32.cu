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
extern "C" __global__ void navatala_ml_compute_effective_dof_f32(const float* eigenvalues, const float* lambda, const unsigned int* p, float* dof) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int pVal = p[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    float eigVal = eigenvalues[gid];
    float denominator = (eigVal + lambdaVal);
    float contribution = (eigVal / denominator);
    sdata[lid] = contribution;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int dof32_reductionStride = 128u;
  for (int dof32_reductionStep = 0; dof32_reductionStep < (int)(8); ++dof32_reductionStep) {
    unsigned int dof32_stride = dof32_reductionStride;
    if ((lid < dof32_stride)) {
      float dof32_other = sdata[(lid + dof32_stride)];
      float dof32_mine = sdata[lid];
      float dof32_sum = (dof32_mine + dof32_other);
      sdata[lid] = dof32_sum;
    }
    unsigned int dof32_strideToHalve = dof32_reductionStride;
    unsigned int dof32_nextStride = (dof32_strideToHalve >> 1u);
    dof32_reductionStride = dof32_nextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float dofVal = sdata[0];
    dof[0] = dofVal;
  }
}
