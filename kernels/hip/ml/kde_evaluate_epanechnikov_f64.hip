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
extern "C" __global__ void navatala_ml_kde_evaluate_epanechnikov_f64(const double* data, const double* queryPoints, const double* bandwidth, const unsigned int* n, const unsigned int* m, double* densities) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numQueries = m[0];
  bool inBounds = (gid < numQueries);
  if (inBounds) {
    unsigned int numSamples = n[0];
    double h = bandwidth[0];
    double x = queryPoints[gid];
    double sum = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(numSamples); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      double xi = data[iU32];
      double diff = (x - xi);
      double u = (diff / h);
      double absU = abs(u);
      double uSq = (u * u);
      double oneMinusUSq = (__longlong_as_double(0x3ff0000000000000ull) - uSq);
      double kernelRaw = (__longlong_as_double(0x3fe8000000000000ull) * oneMinusUSq);
      bool inSupport = (absU <= __longlong_as_double(0x3ff0000000000000ull));
      double kernelVal = ((inSupport) ? (kernelRaw) : (__longlong_as_double(0x0000000000000000ull)));
      double currSum = sum;
      sum = (currSum + kernelVal);
    }
    double nFloat = ((double)(numSamples));
    double nh = (nFloat * h);
    double finalSum = sum;
    double density = (finalSum / nh);
    densities[gid] = density;
  }
}
