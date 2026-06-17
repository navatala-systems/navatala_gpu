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
extern "C" __global__ void navatala_sparse_lanczos_step_f64(const double* Av, const double* vCurrent, const double* vPrevious, const double* alpha, const double* beta, const unsigned int* n, double* vNext, double* betaNext) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((gid < N)) {
    double avi = Av[gid];
    double vc = vCurrent[gid];
    double vp = vPrevious[gid];
    double a = alpha[0];
    double b = beta[0];
    double w = ((avi - (a * vc)) - (b * vp));
    vNext[gid] = w;
    atomicAdd(&betaNext[0], (w * w));
  }
}
