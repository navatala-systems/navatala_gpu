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
__kernel void navatala_ml_ema_update_f64(__global const double* x, __global const double* alpha, __global double* ema, __global uint* count) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    double xVal = x[0];
    double a = alpha[0];
    double oldEma = ema[0];
    uint n = count[0];
    double oneMinusA = (as_double(0x3ff0000000000000ul) - a);
    double alphaX = (a * xVal);
    double omAEma = (oneMinusA * oldEma);
    double emaUpd = (alphaX + omAEma);
    double newEma = (((n == (uint)(0u))) ? (xVal) : (emaUpd));
    ema[0] = newEma;
    uint newCnt = (n + (uint)(1u));
    count[0] = newCnt;
  }
}
