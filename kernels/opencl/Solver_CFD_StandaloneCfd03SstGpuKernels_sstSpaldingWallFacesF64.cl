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
__kernel void Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64(__global const double* tangentialVelocityX, __global const double* tangentialVelocityY, __global const double* tangentialVelocityZ, __global const double* wallDistance, __global const double* kinematicViscosity, __global const double* density, __global const double* dynamicViscosity, __global const double* referenceVelocity, __global const uint* nFaces, __global double* densityOut, __global double* tangentMagnitudeOut, __global double* frictionVelocityOut, __global double* tangentUnitXOut, __global double* tangentUnitYOut, __global double* tangentUnitZOut, __global double* wallShearOnFluidXOut, __global double* wallShearOnFluidYOut, __global double* wallShearOnFluidZOut, __global double* uPlusOut, __global double* yPlusOut, __global double* wallShearMagnitudeOut, __global double* omegaViscousOut, __global double* omegaLogarithmicOut, __global double* omegaWallOut, __global double* turbulentKinematicViscosityWallOut, __global double* finalResidualOut, __global uint* iterationsOut, __global uint* zeroVelocityBranchOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint faceCount = nFaces[0];
  if (gid < faceCount) {
    double velocityX = tangentialVelocityX[gid];
    double velocityY = tangentialVelocityY[gid];
    double velocityZ = tangentialVelocityZ[gid];
    double distance = wallDistance[gid];
    double nu = kinematicViscosity[gid];
    double rho = density[gid];
    double mu = dynamicViscosity[gid];
    double reference = referenceVelocity[gid];
    double tangentMagnitude = sqrt((((velocityX * velocityX) + (velocityY * velocityY)) + (velocityZ * velocityZ)));
    double velocityFloor = (as_double(0x3d06849b86a12b9bul) * (((reference < as_double(0x3ff0000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (reference)));
    double tangentUnitX = (((as_double(0x0000000000000000ul) < tangentMagnitude)) ? ((velocityX / tangentMagnitude)) : (as_double(0x0000000000000000ul)));
    double tangentUnitY = (((as_double(0x0000000000000000ul) < tangentMagnitude)) ? ((velocityY / tangentMagnitude)) : (as_double(0x0000000000000000ul)));
    double tangentUnitZ = (((as_double(0x0000000000000000ul) < tangentMagnitude)) ? ((velocityZ / tangentMagnitude)) : (as_double(0x0000000000000000ul)));
    double omegaViscous = ((as_double(0x4018000000000000ul) * nu) / ((as_double(0x3fb3333333333333ul) * distance) * distance));
    double frictionVelocity = as_double(0x0000000000000000ul);
    double uPlus = as_double(0x0000000000000000ul);
    double yPlus = as_double(0x0000000000000000ul);
    double wallShearMagnitude = as_double(0x0000000000000000ul);
    double omegaLogarithmic = as_double(0x0000000000000000ul);
    double omegaWall = omegaViscous;
    double nutWall = as_double(0x0000000000000000ul);
    double finalResidual = as_double(0x0000000000000000ul);
    uint iterations = (uint)(0u);
    uint zeroVelocityBranch = (uint)(0u);
    if (tangentMagnitude <= velocityFloor) {
      zeroVelocityBranch = (uint)(1u);
    } else {
      double reynoldsY = ((distance * tangentMagnitude) / nu);
      double lower = as_double(0x3eb0000000000000ul);
      double upper = as_double(0x4050000000000000ul);
      double lowerResidual = ((lower + (((((exp((as_double(0x3fda3d70a3d70a3dul) * lower)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * lower)) - (((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / lower));
      for (int iteration = 0; iteration < (int)(80); ++iteration) {
        double midpoint = (lower + (as_double(0x3fe0000000000000ul) * (upper - lower)));
        double midpointResidual = ((midpoint + (((((exp((as_double(0x3fda3d70a3d70a3dul) * midpoint)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * midpoint)) - (((as_double(0x3fda3d70a3d70a3dul) * midpoint) * (as_double(0x3fda3d70a3d70a3dul) * midpoint)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * midpoint) * (as_double(0x3fda3d70a3d70a3dul) * midpoint)) * (as_double(0x3fda3d70a3d70a3dul) * midpoint)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / midpoint));
        if ((midpointResidual < as_double(0x0000000000000000ul)) == (lowerResidual < as_double(0x0000000000000000ul))) {
          lower = midpoint;
          lowerResidual = midpointResidual;
        } else {
          upper = midpoint;
        }
      }
      double midpointCandidate = (lower + (as_double(0x3fe0000000000000ul) * (upper - lower)));
      double selectedUPlus = lower;
      double selectedResidual = (((((lower + (((((exp((as_double(0x3fda3d70a3d70a3dul) * lower)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * lower)) - (((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / lower)) < as_double(0x0000000000000000ul))) ? ((-((lower + (((((exp((as_double(0x3fda3d70a3d70a3dul) * lower)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * lower)) - (((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / lower)))) : (((lower + (((((exp((as_double(0x3fda3d70a3d70a3dul) * lower)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * lower)) - (((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * lower) * (as_double(0x3fda3d70a3d70a3dul) * lower)) * (as_double(0x3fda3d70a3d70a3dul) * lower)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / lower))));
      double upperCandidateResidual = (((((upper + (((((exp((as_double(0x3fda3d70a3d70a3dul) * upper)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * upper)) - (((as_double(0x3fda3d70a3d70a3dul) * upper) * (as_double(0x3fda3d70a3d70a3dul) * upper)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * upper) * (as_double(0x3fda3d70a3d70a3dul) * upper)) * (as_double(0x3fda3d70a3d70a3dul) * upper)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / upper)) < as_double(0x0000000000000000ul))) ? ((-((upper + (((((exp((as_double(0x3fda3d70a3d70a3dul) * upper)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * upper)) - (((as_double(0x3fda3d70a3d70a3dul) * upper) * (as_double(0x3fda3d70a3d70a3dul) * upper)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * upper) * (as_double(0x3fda3d70a3d70a3dul) * upper)) * (as_double(0x3fda3d70a3d70a3dul) * upper)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / upper)))) : (((upper + (((((exp((as_double(0x3fda3d70a3d70a3dul) * upper)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * upper)) - (((as_double(0x3fda3d70a3d70a3dul) * upper) * (as_double(0x3fda3d70a3d70a3dul) * upper)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * upper) * (as_double(0x3fda3d70a3d70a3dul) * upper)) * (as_double(0x3fda3d70a3d70a3dul) * upper)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / upper))));
      if (upperCandidateResidual < selectedResidual) {
        selectedUPlus = upper;
        selectedResidual = upperCandidateResidual;
      }
      double midpointCandidateResidual = (((((midpointCandidate + (((((exp((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) - (((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / midpointCandidate)) < as_double(0x0000000000000000ul))) ? ((-((midpointCandidate + (((((exp((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) - (((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / midpointCandidate)))) : (((midpointCandidate + (((((exp((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) - as_double(0x3ff0000000000000ul)) - (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) - (((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) / as_double(0x4000000000000000ul))) - ((((as_double(0x3fda3d70a3d70a3dul) * midpointCandidate) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) * (as_double(0x3fda3d70a3d70a3dul) * midpointCandidate)) / as_double(0x4018000000000000ul))) / as_double(0x402399999999999aul))) - (reynoldsY / midpointCandidate))));
      if (midpointCandidateResidual < selectedResidual) {
        selectedUPlus = midpointCandidate;
        selectedResidual = midpointCandidateResidual;
      }
      uPlus = selectedUPlus;
      finalResidual = selectedResidual;
      iterations = (uint)(80u);
      frictionVelocity = (tangentMagnitude / selectedUPlus);
      yPlus = ((distance * frictionVelocity) / nu);
      wallShearMagnitude = ((rho * frictionVelocity) * frictionVelocity);
      omegaLogarithmic = (frictionVelocity / ((sqrt(as_double(0x3fb70a3d70a3d70aul)) * as_double(0x3fda3d70a3d70a3dul)) * distance));
      omegaWall = sqrt(((omegaViscous * omegaViscous) + (omegaLogarithmic * omegaLogarithmic)));
      double turbulentDynamicViscosityWall = ((((((((rho * frictionVelocity) * frictionVelocity) * distance) / (((tangentMagnitude < velocityFloor)) ? (velocityFloor) : (tangentMagnitude))) - mu) < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((((((rho * frictionVelocity) * frictionVelocity) * distance) / (((tangentMagnitude < velocityFloor)) ? (velocityFloor) : (tangentMagnitude))) - mu)));
      nutWall = (turbulentDynamicViscosityWall / rho);
    }
    densityOut[gid] = rho;
    tangentMagnitudeOut[gid] = tangentMagnitude;
    frictionVelocityOut[gid] = frictionVelocity;
    tangentUnitXOut[gid] = tangentUnitX;
    tangentUnitYOut[gid] = tangentUnitY;
    tangentUnitZOut[gid] = tangentUnitZ;
    wallShearOnFluidXOut[gid] = (-(wallShearMagnitude * tangentUnitX));
    wallShearOnFluidYOut[gid] = (-(wallShearMagnitude * tangentUnitY));
    wallShearOnFluidZOut[gid] = (-(wallShearMagnitude * tangentUnitZ));
    uPlusOut[gid] = uPlus;
    yPlusOut[gid] = yPlus;
    wallShearMagnitudeOut[gid] = wallShearMagnitude;
    omegaViscousOut[gid] = omegaViscous;
    omegaLogarithmicOut[gid] = omegaLogarithmic;
    omegaWallOut[gid] = omegaWall;
    turbulentKinematicViscosityWallOut[gid] = nutWall;
    finalResidualOut[gid] = finalResidual;
    iterationsOut[gid] = iterations;
    zeroVelocityBranchOut[gid] = zeroVelocityBranch;
  }
}
