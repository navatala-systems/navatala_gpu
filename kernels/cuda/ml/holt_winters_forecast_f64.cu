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
extern "C" __global__ void navatala_ml_holt_winters_forecast_f64(const double* level, const double* trend, const double* season, const unsigned int* period, const unsigned int* batch_size, const unsigned int* horizon, double* forecasts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int horizonVal = horizon[0];
  unsigned int batchSize = batch_size[0];
  unsigned int periodVal = period[0];
  unsigned int totalOutputs = (horizonVal * batchSize);
  bool inBounds = (gid < totalOutputs);
  if (inBounds) {
    unsigned int batch = (gid / horizonVal);
    unsigned int h = ((gid % horizonVal) + 1u);
    double levelVal = level[batch];
    double trendVal = trend[batch];
    unsigned int seasonIdx = ((batch * periodVal) + ((h - 1u) % periodVal));
    double seasonVal = season[seasonIdx];
    double hF = ((double)(h));
    double forecast = ((levelVal + (hF * trendVal)) + seasonVal);
    forecasts[gid] = forecast;
  }
}
