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
extern "C" __global__ void navatala_sparse_amg_cycle_counter_f64(const double* residualNorm, const double* tolerance, const unsigned int* maxCycles, unsigned int* cycleCount, unsigned int* converged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  double norm = residualNorm[0];
  double tol = tolerance[0];
  unsigned int cc = cycleCount[0];
  unsigned int mc = maxCycles[0];
  unsigned int newCc = (cc + 1u);
  unsigned int conv = (((norm < tol)) ? (1u) : (0u));
  unsigned int maxReached = (((newCc >= mc)) ? (1u) : (0u));
  unsigned int flag = ((((conv == 1u) || (maxReached == 1u))) ? (1u) : (0u));
  cycleCount[0] = newCc;
  converged[0] = flag;
}
