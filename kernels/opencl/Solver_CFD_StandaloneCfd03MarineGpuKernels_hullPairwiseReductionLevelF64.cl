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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64(__global const double* pressureForceXIn, __global const double* pressureForceYIn, __global const double* pressureForceZIn, __global const double* viscousForceXIn, __global const double* viscousForceYIn, __global const double* viscousForceZIn, __global const double* pressureMomentXIn, __global const double* pressureMomentYIn, __global const double* pressureMomentZIn, __global const double* viscousMomentXIn, __global const double* viscousMomentYIn, __global const double* viscousMomentZIn, __global const uint* activeCount, __global double* pressureForceXOut, __global double* pressureForceYOut, __global double* pressureForceZOut, __global double* viscousForceXOut, __global double* viscousForceYOut, __global double* viscousForceZOut, __global double* pressureMomentXOut, __global double* pressureMomentYOut, __global double* pressureMomentZOut, __global double* viscousMomentXOut, __global double* viscousMomentYOut, __global double* viscousMomentZOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint _active = activeCount[0];
  uint activeHalf = (_active / (uint)(2u));
  uint outputCount = (activeHalf + (_active % (uint)(2u)));
  if (gid < outputCount) {
    uint leftIndex = (gid * (uint)(2u));
    uint rightIndex = (leftIndex + (uint)(1u));
    bool hasRight = (rightIndex < _active);
    double pressureForceXLeft = pressureForceXIn[leftIndex];
    double pressureForceYLeft = pressureForceYIn[leftIndex];
    double pressureForceZLeft = pressureForceZIn[leftIndex];
    double viscousForceXLeft = viscousForceXIn[leftIndex];
    double viscousForceYLeft = viscousForceYIn[leftIndex];
    double viscousForceZLeft = viscousForceZIn[leftIndex];
    double pressureMomentXLeft = pressureMomentXIn[leftIndex];
    double pressureMomentYLeft = pressureMomentYIn[leftIndex];
    double pressureMomentZLeft = pressureMomentZIn[leftIndex];
    double viscousMomentXLeft = viscousMomentXIn[leftIndex];
    double viscousMomentYLeft = viscousMomentYIn[leftIndex];
    double viscousMomentZLeft = viscousMomentZIn[leftIndex];
    pressureForceXOut[gid] = ((hasRight) ? ((pressureForceXLeft + pressureForceXIn[rightIndex])) : (pressureForceXLeft));
    pressureForceYOut[gid] = ((hasRight) ? ((pressureForceYLeft + pressureForceYIn[rightIndex])) : (pressureForceYLeft));
    pressureForceZOut[gid] = ((hasRight) ? ((pressureForceZLeft + pressureForceZIn[rightIndex])) : (pressureForceZLeft));
    viscousForceXOut[gid] = ((hasRight) ? ((viscousForceXLeft + viscousForceXIn[rightIndex])) : (viscousForceXLeft));
    viscousForceYOut[gid] = ((hasRight) ? ((viscousForceYLeft + viscousForceYIn[rightIndex])) : (viscousForceYLeft));
    viscousForceZOut[gid] = ((hasRight) ? ((viscousForceZLeft + viscousForceZIn[rightIndex])) : (viscousForceZLeft));
    pressureMomentXOut[gid] = ((hasRight) ? ((pressureMomentXLeft + pressureMomentXIn[rightIndex])) : (pressureMomentXLeft));
    pressureMomentYOut[gid] = ((hasRight) ? ((pressureMomentYLeft + pressureMomentYIn[rightIndex])) : (pressureMomentYLeft));
    pressureMomentZOut[gid] = ((hasRight) ? ((pressureMomentZLeft + pressureMomentZIn[rightIndex])) : (pressureMomentZLeft));
    viscousMomentXOut[gid] = ((hasRight) ? ((viscousMomentXLeft + viscousMomentXIn[rightIndex])) : (viscousMomentXLeft));
    viscousMomentYOut[gid] = ((hasRight) ? ((viscousMomentYLeft + viscousMomentYIn[rightIndex])) : (viscousMomentYLeft));
    viscousMomentZOut[gid] = ((hasRight) ? ((viscousMomentZLeft + viscousMomentZIn[rightIndex])) : (viscousMomentZLeft));
  }
}
