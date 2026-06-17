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

__kernel void navatala_dataframe_covariance_update_f32(__global const float* x, __global const float* y, __global uint* count, __global float* meanX, __global float* meanY, __global float* Cxy) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    uint nOld = count[0];
    float muXOld = meanX[0];
    float muYOld = meanY[0];
    float cxyOld = Cxy[0];
    float xVal = x[0];
    float yVal = y[0];
    uint nNew = (nOld + (uint)(1u));
    float nNewF = ((float)(nNew));
    float deltaX = (xVal - muXOld);
    float deltaY = (yVal - muYOld);
    float muXNew = (muXOld + (deltaX / nNewF));
    float muYNew = (muYOld + (deltaY / nNewF));
    float yMinusMuYNew = (yVal - muYNew);
    float cxyNew = (cxyOld + (deltaX * yMinusMuYNew));
    count[0] = nNew;
    meanX[0] = muXNew;
    meanY[0] = muYNew;
    Cxy[0] = cxyNew;
  }
}
