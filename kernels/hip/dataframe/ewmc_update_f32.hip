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
extern "C" __global__ void navatala_dataframe_ewmc_update_f32(const float* x, const float* y, const float* alpha, float* nEff, float* meanX, float* meanY, float* Cxy, float* varX, float* varY) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    float xVal = x[0];
    float yVal = y[0];
    float a = alpha[0];
    float oldNEff = nEff[0];
    float oldMeanX = meanX[0];
    float oldMeanY = meanY[0];
    float oldCxy = Cxy[0];
    float oldVarX = varX[0];
    float oldVarY = varY[0];
    float omA = (__uint_as_float(0x3f800000u) - a);
    float omANEff = (omA * oldNEff);
    float newNEff = (a + omANEff);
    float deltaX = (xVal - oldMeanX);
    float deltaY = (yVal - oldMeanY);
    float aDeltaX = (a * deltaX);
    float newMeanX = (oldMeanX + aDeltaX);
    float aDeltaY = (a * deltaY);
    float newMeanY = (oldMeanY + aDeltaY);
    float deltaXNew = (xVal - newMeanX);
    float deltaYNew = (yVal - newMeanY);
    float dxDyNew = (deltaX * deltaYNew);
    float aDxDyNew = (a * dxDyNew);
    float cxySum = (oldCxy + aDxDyNew);
    float newCxy = (omA * cxySum);
    float dxDxNew = (deltaX * deltaXNew);
    float aDxDxNew = (a * dxDxNew);
    float varXSum = (oldVarX + aDxDxNew);
    float newVarX = (omA * varXSum);
    float dyDyNew = (deltaY * deltaYNew);
    float aDyDyNew = (a * dyDyNew);
    float varYSum = (oldVarY + aDyDyNew);
    float newVarY = (omA * varYSum);
    nEff[0] = newNEff;
    meanX[0] = newMeanX;
    meanY[0] = newMeanY;
    Cxy[0] = newCxy;
    varX[0] = newVarX;
    varY[0] = newVarY;
  }
}
