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
__kernel void navatala_cfd_stf_all_faces_f64(__global const double* divN, __global const float* alpha, __global const float* alphaF, __global const float* delta, __global const int* owner, __global const int* nei, __global const float* weights, __global const double* contactGrad, __global const uint* thetaMask, __global const float* paramsF, __global const int* counts, __global double* stf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    double w = ((double)(weights[((int)((int)(get_global_id(0))))]));
    double wm = (as_double(0x3ff0000000000000ul) - w);
    double divInterp = ((w * divN[o]) + (wm * divN[n]));
    double Kf = (-divInterp);
    double snGrad = as_double(0x0000000000000000ul);
    if (((int)((int)(get_global_id(0)))) < counts[2]) {
      snGrad = (((double)(delta[((int)((int)(get_global_id(0))))])) * (((double)(alpha[n])) - ((double)(alpha[o]))));
    } else {
      if (thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u)) {
        snGrad = contactGrad[((int)((int)(get_global_id(0))))];
      } else {
        snGrad = (((double)(delta[((int)((int)(get_global_id(0))))])) * (((double)(alphaF[((int)((int)(get_global_id(0))))])) - ((double)(alpha[o]))));
      }
    }
    stf[((int)((int)(get_global_id(0))))] = ((((double)(paramsF[0])) * Kf) * snGrad);
  }
}
