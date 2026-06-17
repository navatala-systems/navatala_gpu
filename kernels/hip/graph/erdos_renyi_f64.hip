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
extern "C" __global__ void navatala_graph_erdos_renyi_f64(const double* random, const double* randomWeights, const double* p, const unsigned int* n, unsigned int* srcNodes, unsigned int* dstNodes, double* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0u];
  unsigned int nMinus1 = (numN - 1u);
  unsigned int totalPairs = ((numN * nMinus1) / 2u);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    double prob = p[0u];
    double gidF = ((double)(gid));
    double discriminant = (__longlong_as_double(0x3ff0000000000000ull) + (__longlong_as_double(0x4020000000000000ull) * gidF));
    double sqrtDisc = sqrt(discriminant);
    double iFloat = ((__longlong_as_double(0x3ff0000000000000ull) + sqrtDisc) / __longlong_as_double(0x4000000000000000ull));
    unsigned int i = ((unsigned int)(floor(iFloat)));
    unsigned int iTriangle = ((i * (i - 1u)) / 2u);
    unsigned int j = (gid - iTriangle);
    double r = random[gid];
    bool includeEdge = (r < prob);
    if (includeEdge) {
      double w = randomWeights[gid];
      unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
      srcNodes[slot] = j;
      dstNodes[slot] = i;
      weights[slot] = w;
    }
  }
}
