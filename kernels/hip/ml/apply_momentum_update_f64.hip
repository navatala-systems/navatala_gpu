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
extern "C" __global__ void navatala_ml_apply_momentum_update_f64(const double* Y, const double* Y_prev, const double* grad, const double* lr, const double* momentum, const unsigned int* nPoints, const unsigned int* outputDims, double* Y_new) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = outputDims[0];
  unsigned int totalElems = (n * d);
  double lrVal = lr[0];
  double momVal = momentum[0];
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    double y = Y[gid];
    double yPrev = Y_prev[gid];
    double g = grad[gid];
    double lrGrad = (lrVal * g);
    double yDiff = (y - yPrev);
    double momTerm = (momVal * yDiff);
    double yNew = (y + (lrGrad + momTerm));
    Y_new[gid] = yNew;
  }
}
