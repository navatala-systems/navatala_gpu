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
extern "C" __global__ void navatala_graph_normalized_laplacian_f64(const double* W, const double* D, const unsigned int* n, double* L_sym) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    double di = D[i];
    double dj = D[j];
    double eps = __longlong_as_double(0x3cd203af9ee75616ull);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = __longlong_as_double(0x3ff0000000000000ull);
      } else {
        L_sym[gid] = __longlong_as_double(0x0000000000000000ull);
      }
    } else {
      double wVal = W[gid];
      double didjProd = (di * dj);
      double sqrtDiDj = sqrt(didjProd);
      double normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        double lSymVal = (__longlong_as_double(0x3ff0000000000000ull) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        double lSymVal = (__longlong_as_double(0x0000000000000000ull) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}
