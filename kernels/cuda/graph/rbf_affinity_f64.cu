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
extern "C" __global__ void navatala_graph_rbf_affinity_f64(const double* data, const double* sigma, const unsigned int* n, const unsigned int* d, double* W) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int dVal = d[0];
  double sigmaVal = sigma[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = __longlong_as_double(0x0000000000000000ull);
    } else {
      double distSq = __longlong_as_double(0x0000000000000000ull);
      unsigned int rowI = (i * dVal);
      unsigned int rowJ = (j * dVal);
      unsigned int idxI = rowI;
      unsigned int idxJ = rowJ;
      double xi = data[idxI];
      double xj = data[idxJ];
      double diff = (xi - xj);
      double diffSq = (diff * diff);
      double sigmaSq = (sigmaVal * sigmaVal);
      double twoSigmaSq = (__longlong_as_double(0x4000000000000000ull) * sigmaSq);
      double exponent = ((__longlong_as_double(0x0000000000000000ull) - diffSq) / twoSigmaSq);
      double affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}
