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
__kernel void navatala_cfd_mules_proc_face_update_f64(__global const int* procFaces, __global const int* counts, __global const double* phiCorr, __global double* lambda, __global const double* lambdapCell, __global const double* lambdamCell, __global const int* ownerAll, __global const double* nbrLambdaP, __global const double* nbrLambdaM) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int face = procFaces[((int)((int)(get_global_id(0))))];
    int o = ownerAll[face];
    double pc = phiCorr[face];
    double lim = lambdamCell[o];
    double other = nbrLambdaP[((int)((int)(get_global_id(0))))];
    if (lim > other) {
      lim = other;
    }
    if (pc > as_double(0x0000000000000000ul)) {
      double limPos = lambdapCell[o];
      double otherPos = nbrLambdaM[((int)((int)(get_global_id(0))))];
      if (limPos > otherPos) {
        limPos = otherPos;
      }
      lim = limPos;
    }
    double cur = lambda[face];
    double _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[face] = _out;
  }
}
