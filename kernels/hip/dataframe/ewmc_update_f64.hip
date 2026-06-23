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
extern "C" __global__ void navatala_dataframe_ewmc_update_f64(const double* x, const double* y, const double* alpha, double* nEff, double* meanX, double* meanY, double* Cxy, double* varX, double* varY) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    double xVal = x[0];
    double yVal = y[0];
    double a = alpha[0];
    double oldNEff = nEff[0];
    double oldMeanX = meanX[0];
    double oldMeanY = meanY[0];
    double oldCxy = Cxy[0];
    double oldVarX = varX[0];
    double oldVarY = varY[0];
    double omA = (__longlong_as_double(0x3ff0000000000000ull) - a);
    double omANEff = (omA * oldNEff);
    double newNEff = (a + omANEff);
    double deltaX = (xVal - oldMeanX);
    double deltaY = (yVal - oldMeanY);
    double aDeltaX = (a * deltaX);
    double newMeanX = (oldMeanX + aDeltaX);
    double aDeltaY = (a * deltaY);
    double newMeanY = (oldMeanY + aDeltaY);
    double deltaXNew = (xVal - newMeanX);
    double deltaYNew = (yVal - newMeanY);
    double dxDyNew = (deltaX * deltaYNew);
    double aDxDyNew = (a * dxDyNew);
    double cxySum = (oldCxy + aDxDyNew);
    double newCxy = (omA * cxySum);
    double dxDxNew = (deltaX * deltaXNew);
    double aDxDxNew = (a * dxDxNew);
    double varXSum = (oldVarX + aDxDxNew);
    double newVarX = (omA * varXSum);
    double dyDyNew = (deltaY * deltaYNew);
    double aDyDyNew = (a * dyDyNew);
    double varYSum = (oldVarY + aDyDyNew);
    double newVarY = (omA * varYSum);
    nEff[0] = newNEff;
    meanX[0] = newMeanX;
    meanY[0] = newMeanY;
    Cxy[0] = newCxy;
    varX[0] = newVarX;
    varY[0] = newVarY;
  }
}
