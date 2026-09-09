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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64(__global const double* density, __global const double* volume, __global const double* streamwiseCoordinate, __global const double* acceptedVelocityX, __global const double* acceptedVelocityY, __global const double* acceptedVelocityZ, __global const double* dampingDatum, __global const uint* nCells, __global double* lambdaOut, __global double* implicitDiagonalOut, __global double* implicitTargetRhsXOut, __global double* implicitTargetRhsYOut, __global double* implicitTargetRhsZOut, __global double* momentumSourceXOut, __global double* momentumSourceYOut, __global double* momentumSourceZOut, __global double* signedPowerOut, __global double* signedWorkOut, __global uint* activeOut) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int cellCount = ((int)(nCells[0]));
  if (gid < cellCount) {
    double rawXi = ((streamwiseCoordinate[gid] - dampingDatum[1]) / (dampingDatum[2] - dampingDatum[1]));
    double xi = (((rawXi < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((((as_double(0x3ff0000000000000ul) < rawXi)) ? (as_double(0x3ff0000000000000ul)) : (rawXi))));
    double ramp = ((xi * xi) * (as_double(0x4008000000000000ul) - (as_double(0x4000000000000000ul) * xi)));
    double lambda = (dampingDatum[0] * ramp);
    double scale = ((density[gid] * lambda) * volume[gid]);
    double targetRhsX = (scale * dampingDatum[3]);
    double targetRhsY = (scale * dampingDatum[4]);
    double targetRhsZ = (scale * dampingDatum[5]);
    double sourceX = (targetRhsX - (scale * acceptedVelocityX[gid]));
    double sourceY = (targetRhsY - (scale * acceptedVelocityY[gid]));
    double sourceZ = (targetRhsZ - (scale * acceptedVelocityZ[gid]));
    double signedPower = (((acceptedVelocityX[gid] * sourceX) + (acceptedVelocityY[gid] * sourceY)) + (acceptedVelocityZ[gid] * sourceZ));
    double signedWork = (dampingDatum[6] * signedPower);
    lambdaOut[gid] = lambda;
    implicitDiagonalOut[gid] = scale;
    implicitTargetRhsXOut[gid] = targetRhsX;
    implicitTargetRhsYOut[gid] = targetRhsY;
    implicitTargetRhsZOut[gid] = targetRhsZ;
    momentumSourceXOut[gid] = sourceX;
    momentumSourceYOut[gid] = sourceY;
    momentumSourceZOut[gid] = sourceZ;
    signedPowerOut[gid] = signedPower;
    signedWorkOut[gid] = signedWork;
    activeOut[gid] = (((as_double(0x0000000000000000ul) < lambda)) ? ((uint)(1u)) : ((uint)(0u)));
  }
}
