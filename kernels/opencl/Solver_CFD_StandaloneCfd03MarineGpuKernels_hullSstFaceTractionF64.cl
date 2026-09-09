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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64(__global const double* wallShearOnFluidX, __global const double* wallShearOnFluidY, __global const double* wallShearOnFluidZ, __global const double* fluidAreaVectorX, __global const double* fluidAreaVectorY, __global const double* fluidAreaVectorZ, __global const double* faceCentroidX, __global const double* faceCentroidY, __global const double* faceCentroidZ, __global const double* hullSstDatum, __global const uint* nFaces, __global double* areaOut, __global double* forceXOut, __global double* forceYOut, __global double* forceZOut, __global double* momentXOut, __global double* momentYOut, __global double* momentZOut) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double forceX = ((-wallShearOnFluidX[gid]) * area);
    double forceY = ((-wallShearOnFluidY[gid]) * area);
    double forceZ = ((-wallShearOnFluidZ[gid]) * area);
    double momentArmX = (faceCentroidX[gid] - hullSstDatum[0]);
    double momentArmY = (faceCentroidY[gid] - hullSstDatum[1]);
    double momentArmZ = (faceCentroidZ[gid] - hullSstDatum[2]);
    double momentX = ((momentArmY * forceZ) - (momentArmZ * forceY));
    double momentY = ((momentArmZ * forceX) - (momentArmX * forceZ));
    double momentZ = ((momentArmX * forceY) - (momentArmY * forceX));
    areaOut[gid] = area;
    forceXOut[gid] = forceX;
    forceYOut[gid] = forceY;
    forceZOut[gid] = forceZ;
    momentXOut[gid] = momentX;
    momentYOut[gid] = momentY;
    momentZOut[gid] = momentZ;
  }
}
