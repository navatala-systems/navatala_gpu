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
extern "C" __global__ void navatala_ml_kde_bandwidth_silverman_f64(const double* stddev, const unsigned int* n, double* bandwidth) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gid == 0u)) {
    double sigma = stddev[0];
    unsigned int count = n[0];
    double countF = ((double)(count));
    double logN = log(countF);
    double negLogN = (__longlong_as_double(0xbfc999999999999aull) * logN);
    double nPowNeg02 = exp(negLogN);
    double h = (__longlong_as_double(0x3ff0f5c28f5c28f6ull) * (sigma * nPowNeg02));
    bandwidth[0] = h;
  }
}
