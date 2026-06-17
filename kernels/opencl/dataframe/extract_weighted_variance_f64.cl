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
__kernel void navatala_dataframe_extract_weighted_variance_f64(__global const double* weightSum, __global const double* M2, __global double* variance) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    double w = weightSum[0];
    double m2Val = M2[0];
    double var = (m2Val / w);
    variance[0] = var;
  }
}
