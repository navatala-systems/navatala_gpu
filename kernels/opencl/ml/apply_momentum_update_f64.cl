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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_apply_momentum_update_f64(__global const double* Y, __global const double* Y_prev, __global const double* grad, __global const double* lr, __global const double* momentum, __global const uint* nPoints, __global const uint* outputDims, __global double* Y_new) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalElems = (n * d);
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
