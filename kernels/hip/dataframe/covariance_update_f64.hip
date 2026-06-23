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
extern "C" __global__ void navatala_dataframe_covariance_update_f64(const double* x, const double* y, unsigned int* count, double* meanX, double* meanY, double* Cxy) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    unsigned int nOld = count[0];
    double muXOld = meanX[0];
    double muYOld = meanY[0];
    double cxyOld = Cxy[0];
    double xVal = x[0];
    double yVal = y[0];
    unsigned int nNew = (nOld + 1u);
    double nNewF = ((double)(nNew));
    double deltaX = (xVal - muXOld);
    double deltaY = (yVal - muYOld);
    double muXNew = (muXOld + (deltaX / nNewF));
    double muYNew = (muYOld + (deltaY / nNewF));
    double yMinusMuYNew = (yVal - muYNew);
    double cxyNew = (cxyOld + (deltaX * yMinusMuYNew));
    count[0] = nNew;
    meanX[0] = muXNew;
    meanY[0] = muYNew;
    Cxy[0] = cxyNew;
  }
}
