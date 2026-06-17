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

__kernel void navatala_ml_holt_winters_forecast_f32(__global const float* level, __global const float* trend, __global const float* season, __global const uint* period, __global const uint* batch_size, __global const uint* horizon, __global float* forecasts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint horizonVal = horizon[0];
  uint batchSize = batch_size[0];
  uint periodVal = period[0];
  uint totalOutputs = (horizonVal * batchSize);
  bool inBounds = (gid < totalOutputs);
  if (inBounds) {
    uint batch = (gid / horizonVal);
    uint h = ((gid % horizonVal) + (uint)(1u));
    float levelVal = level[batch];
    float trendVal = trend[batch];
    uint seasonIdx = ((batch * periodVal) + ((h - (uint)(1u)) % periodVal));
    float seasonVal = season[seasonIdx];
    float hF = ((float)(h));
    float forecast = ((levelVal + (hF * trendVal)) + seasonVal);
    forecasts[gid] = forecast;
  }
}
