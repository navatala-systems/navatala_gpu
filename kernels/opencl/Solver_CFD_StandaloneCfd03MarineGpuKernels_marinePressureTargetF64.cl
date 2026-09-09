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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64(__global const double* pointX, __global const double* pointY, __global const double* pointZ, __global const double* liquidFraction, __global const double* datum, __global const uint* nCells, __global double* densityOut, __global double* pRghOut, __global double* absolutePressureOut) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int cellCount = ((int)(nCells[0]));
  if (gid < cellCount) {
    double alpha = liquidFraction[gid];
    double density = ((alpha * datum[10]) + ((as_double(0x3ff0000000000000ul) - alpha) * datum[11]));
    double datumGravityHead = (((datum[7] * (datum[1] - datum[4])) + (datum[8] * (datum[2] - datum[5]))) + (datum[9] * (datum[3] - datum[6])));
    double pRgh = (datum[0] + (density * datumGravityHead));
    double pointGravityHead = (((datum[7] * (pointX[gid] - datum[1])) + (datum[8] * (pointY[gid] - datum[2]))) + (datum[9] * (pointZ[gid] - datum[3])));
    double absolutePressure = (pRgh + (density * pointGravityHead));
    densityOut[gid] = density;
    pRghOut[gid] = pRgh;
    absolutePressureOut[gid] = absolutePressure;
  }
}
