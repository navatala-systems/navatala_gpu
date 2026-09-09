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

#include <hip/hip_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64(const double* tangentialVelocityX, const double* tangentialVelocityY, const double* tangentialVelocityZ, const double* wallDistance, const double* kinematicViscosity, const double* density, const double* dynamicViscosity, const double* referenceVelocity, const unsigned int* nFaces, double* densityOut, double* tangentMagnitudeOut, double* frictionVelocityOut, double* tangentUnitXOut, double* tangentUnitYOut, double* tangentUnitZOut, double* wallShearOnFluidXOut, double* wallShearOnFluidYOut, double* wallShearOnFluidZOut, double* uPlusOut, double* yPlusOut, double* wallShearMagnitudeOut, double* omegaViscousOut, double* omegaLogarithmicOut, double* omegaWallOut, double* turbulentKinematicViscosityWallOut, double* finalResidualOut, unsigned int* iterationsOut, unsigned int* zeroVelocityBranchOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int faceCount = nFaces[0];
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
    double velocityFloor = (__longlong_as_double(0x3d06849b86a12b9bull) * (((reference < __longlong_as_double(0x3ff0000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (reference)));
    double tangentUnitX = (((__longlong_as_double(0x0000000000000000ull) < tangentMagnitude)) ? ((velocityX / tangentMagnitude)) : (__longlong_as_double(0x0000000000000000ull)));
    double tangentUnitY = (((__longlong_as_double(0x0000000000000000ull) < tangentMagnitude)) ? ((velocityY / tangentMagnitude)) : (__longlong_as_double(0x0000000000000000ull)));
    double tangentUnitZ = (((__longlong_as_double(0x0000000000000000ull) < tangentMagnitude)) ? ((velocityZ / tangentMagnitude)) : (__longlong_as_double(0x0000000000000000ull)));
    double omegaViscous = ((__longlong_as_double(0x4018000000000000ull) * nu) / ((__longlong_as_double(0x3fb3333333333333ull) * distance) * distance));
    double frictionVelocity = __longlong_as_double(0x0000000000000000ull);
    double uPlus = __longlong_as_double(0x0000000000000000ull);
    double yPlus = __longlong_as_double(0x0000000000000000ull);
    double wallShearMagnitude = __longlong_as_double(0x0000000000000000ull);
    double omegaLogarithmic = __longlong_as_double(0x0000000000000000ull);
    double omegaWall = omegaViscous;
    double nutWall = __longlong_as_double(0x0000000000000000ull);
    double finalResidual = __longlong_as_double(0x0000000000000000ull);
    unsigned int iterations = 0u;
    unsigned int zeroVelocityBranch = 0u;
    if (tangentMagnitude <= velocityFloor) {
      zeroVelocityBranch = 1u;
    } else {
      double reynoldsY = ((distance * tangentMagnitude) / nu);
      double lower = __longlong_as_double(0x3eb0000000000000ull);
      double upper = __longlong_as_double(0x4050000000000000ull);
      double lowerResidual = ((lower + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / lower));
      for (int iteration = 0; iteration < (int)(80); ++iteration) {
        double midpoint = (lower + (__longlong_as_double(0x3fe0000000000000ull) * (upper - lower)));
        double midpointResidual = ((midpoint + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpoint)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / midpoint));
        if ((midpointResidual < __longlong_as_double(0x0000000000000000ull)) == (lowerResidual < __longlong_as_double(0x0000000000000000ull))) {
          lower = midpoint;
          lowerResidual = midpointResidual;
        } else {
          upper = midpoint;
        }
      }
      double midpointCandidate = (lower + (__longlong_as_double(0x3fe0000000000000ull) * (upper - lower)));
      double selectedUPlus = lower;
      double selectedResidual = (((((lower + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / lower)) < __longlong_as_double(0x0000000000000000ull))) ? ((-((lower + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / lower)))) : (((lower + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * lower) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * lower)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / lower))));
      double upperCandidateResidual = (((((upper + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / upper)) < __longlong_as_double(0x0000000000000000ull))) ? ((-((upper + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / upper)))) : (((upper + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * upper) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * upper)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / upper))));
      if (upperCandidateResidual < selectedResidual) {
        selectedUPlus = upper;
        selectedResidual = upperCandidateResidual;
      }
      double midpointCandidateResidual = (((((midpointCandidate + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / midpointCandidate)) < __longlong_as_double(0x0000000000000000ull))) ? ((-((midpointCandidate + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / midpointCandidate)))) : (((midpointCandidate + (((((exp((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) - __longlong_as_double(0x3ff0000000000000ull)) - (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) - (((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) / __longlong_as_double(0x4000000000000000ull))) - ((((__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) * (__longlong_as_double(0x3fda3d70a3d70a3dull) * midpointCandidate)) / __longlong_as_double(0x4018000000000000ull))) / __longlong_as_double(0x402399999999999aull))) - (reynoldsY / midpointCandidate))));
      if (midpointCandidateResidual < selectedResidual) {
        selectedUPlus = midpointCandidate;
        selectedResidual = midpointCandidateResidual;
      }
      uPlus = selectedUPlus;
      finalResidual = selectedResidual;
      iterations = 80u;
      frictionVelocity = (tangentMagnitude / selectedUPlus);
      yPlus = ((distance * frictionVelocity) / nu);
      wallShearMagnitude = ((rho * frictionVelocity) * frictionVelocity);
      omegaLogarithmic = (frictionVelocity / ((sqrt(__longlong_as_double(0x3fb70a3d70a3d70aull)) * __longlong_as_double(0x3fda3d70a3d70a3dull)) * distance));
      omegaWall = sqrt(((omegaViscous * omegaViscous) + (omegaLogarithmic * omegaLogarithmic)));
      double turbulentDynamicViscosityWall = ((((((((rho * frictionVelocity) * frictionVelocity) * distance) / (((tangentMagnitude < velocityFloor)) ? (velocityFloor) : (tangentMagnitude))) - mu) < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((((rho * frictionVelocity) * frictionVelocity) * distance) / (((tangentMagnitude < velocityFloor)) ? (velocityFloor) : (tangentMagnitude))) - mu)));
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
