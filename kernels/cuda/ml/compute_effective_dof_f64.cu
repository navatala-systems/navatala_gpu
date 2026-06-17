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
extern "C" __global__ void navatala_ml_compute_effective_dof_f64(const double* eigenvalues, const double* lambda, const unsigned int* p, double* dof) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int pVal = p[0];
  double lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    double eigVal = eigenvalues[gid];
    double denominator = (eigVal + lambdaVal);
    double contribution = (eigVal / denominator);
    sdata[lid] = contribution;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int dof64_reductionStride = 128u;
  for (int dof64_reductionStep = 0; dof64_reductionStep < (int)(8); ++dof64_reductionStep) {
    unsigned int dof64_stride = dof64_reductionStride;
    if ((lid < dof64_stride)) {
      double dof64_other = sdata[(lid + dof64_stride)];
      double dof64_mine = sdata[lid];
      double dof64_sum = (dof64_mine + dof64_other);
      sdata[lid] = dof64_sum;
    }
    unsigned int dof64_strideToHalve = dof64_reductionStride;
    unsigned int dof64_nextStride = (dof64_strideToHalve >> 1u);
    dof64_reductionStride = dof64_nextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double dofVal = sdata[0];
    dof[0] = dofVal;
  }
}
