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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_covariance_update_f32(const float* x, const float* y, unsigned int* count, float* meanX, float* meanY, float* Cxy) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    unsigned int nOld = count[0];
    float muXOld = meanX[0];
    float muYOld = meanY[0];
    float cxyOld = Cxy[0];
    float xVal = x[0];
    float yVal = y[0];
    unsigned int nNew = (nOld + 1u);
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
