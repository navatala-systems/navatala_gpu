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

__kernel void navatala_ml_apply_momentum_update_f32(__global const float* Y, __global const float* Y_prev, __global const float* grad, __global const float* lr, __global const float* momentum, __global const uint* nPoints, __global const uint* outputDims, __global float* Y_new) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalElems = (n * d);
  float lrVal = lr[0];
  float momVal = momentum[0];
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    float y = Y[gid];
    float yPrev = Y_prev[gid];
    float g = grad[gid];
    float lrGrad = (lrVal * g);
    float yDiff = (y - yPrev);
    float momTerm = (momVal * yDiff);
    float yNew = (y + (lrGrad + momTerm));
    Y_new[gid] = yNew;
  }
}
