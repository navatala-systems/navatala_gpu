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

__kernel void navatala_ml_ar_predict_f32(__global const float* y, __global const float* phi, __global const float* intercept, __global const uint* p, __global const uint* batch_size, __global float* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint batchSize = batch_size[0];
  uint pVal = p[0];
  bool inBounds = (gid < batchSize);
  if (inBounds) {
    float pred = intercept[gid];
    float sumAR = pred;
    for (int i = 0; i < (int)(pVal); ++i) {
      uint iU32 = ((uint)(i));
      uint yIdx = ((gid * pVal) + iU32);
      float yVal = y[yIdx];
      uint phiIdx = yIdx;
      float phiVal = phi[phiIdx];
      float term = (phiVal * yVal);
      float currSum = sumAR;
      float newSum = (currSum + term);
      sumAR = newSum;
    }
    float finalPred = sumAR;
    predictions[gid] = finalPred;
  }
}
