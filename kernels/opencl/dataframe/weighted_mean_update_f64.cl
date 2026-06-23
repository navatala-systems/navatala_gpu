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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_dataframe_weighted_mean_update_f64(__global const double* x, __global const double* w, __global double* weightSum, __global double* mean) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    double wOld = weightSum[0];
    double meanOld = mean[0];
    double xVal = x[0];
    double wVal = w[0];
    double wNew = (wOld + wVal);
    double wNewSafe = (((wNew == as_double(0x0000000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (wNew));
    double delta = (xVal - meanOld);
    double meanNew = (meanOld + ((wVal / wNewSafe) * delta));
    weightSum[0] = wNew;
    mean[0] = meanNew;
  }
}
