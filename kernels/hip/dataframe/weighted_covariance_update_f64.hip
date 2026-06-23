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
extern "C" __global__ void navatala_dataframe_weighted_covariance_update_f64(const double* x, const double* y, const double* w, double* weightSum, double* meanX, double* meanY, double* Cxy) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    double wOld = weightSum[0];
    double muXOld = meanX[0];
    double muYOld = meanY[0];
    double cxyOld = Cxy[0];
    double xVal = x[0];
    double yVal = y[0];
    double wVal = w[0];
    double wNew = (wOld + wVal);
    double wNewSafe = (((wNew == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (wNew));
    double deltaX = (xVal - muXOld);
    double deltaY = (yVal - muYOld);
    double muXNew = (muXOld + ((wVal / wNewSafe) * deltaX));
    double muYNew = (muYOld + ((wVal / wNewSafe) * deltaY));
    double yMinusMuYNew = (yVal - muYNew);
    double cxyNew = (cxyOld + (wVal * (deltaX * yMinusMuYNew)));
    weightSum[0] = wNew;
    meanX[0] = muXNew;
    meanY[0] = muYNew;
    Cxy[0] = cxyNew;
  }
}
