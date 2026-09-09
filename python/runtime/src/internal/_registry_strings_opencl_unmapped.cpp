// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_unmapped` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64(__global const ulong* probeOffsets, __global const double* verticalCoordinates, __global const double* alphaValues, __global const uint* nProbes, __global double* elevationOut, __global ulong* lowerPointOut, __global uint* crossingCountOut, __global uint* exactPointOut, __global uint* crossingValidOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint probeCount = nProbes[0];
  if (gid < probeCount) {
    uint nextProbe = (gid + (uint)(1u));
    ulong pointBegin = probeOffsets[gid];
    ulong pointEnd = probeOffsets[nextProbe];
    ulong pointCount = (pointEnd - pointBegin);
    ulong segmentCount = (pointCount - ((ulong)(1)));
    uint crossingCount = (uint)(0u);
    ulong exactIndex = ((ulong)(-1));
    double elevation = as_double(0x0000000000000000ul);
    ulong lowerPoint = ((ulong)(-1));
    uint exactPoint = (uint)(0u);
    for (int point = 0; point < (int)(pointCount); ++point) {
      ulong pointLocal = ((ulong)(point));
      ulong pointFlat = (pointBegin + pointLocal);
      double pointAlpha = alphaValues[pointFlat];
      if (pointAlpha == as_double(0x3fe0000000000000ul)) {
        uint exactCrossingCount = (crossingCount + (uint)(1u));
        crossingCount = exactCrossingCount;
        exactIndex = pointLocal;
      }
    }
    for (int segment = 0; segment < (int)(segmentCount); ++segment) {
      ulong segmentLocal = ((ulong)(segment));
      ulong segmentFlat = (pointBegin + segmentLocal);
      ulong nextSegmentFlat = (segmentFlat + ((ulong)(1)));
      double coordinateA = verticalCoordinates[segmentFlat];
      double coordinateB = verticalCoordinates[nextSegmentFlat];
      double a = (alphaValues[segmentFlat] - as_double(0x3fe0000000000000ul));
      double b = (alphaValues[nextSegmentFlat] - as_double(0x3fe0000000000000ul));
      if (a < as_double(0x0000000000000000ul)) {
        if (as_double(0x0000000000000000ul) < b) {
          uint strictCrossingCount = (crossingCount + (uint)(1u));
          crossingCount = strictCrossingCount;
          lowerPoint = segmentLocal;
          if (strictCrossingCount == (uint)(1u)) {
            elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
          }
        }
      } else {
        if (as_double(0x0000000000000000ul) < a) {
          if (b < as_double(0x0000000000000000ul)) {
            uint strictCrossingCount = (crossingCount + (uint)(1u));
            crossingCount = strictCrossingCount;
            lowerPoint = segmentLocal;
            if (strictCrossingCount == (uint)(1u)) {
              elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
            }
          }
        }
      }
    }
    uint crossingValid = (((crossingCount == (uint)(1u))) ? ((uint)(1u)) : ((uint)(0u)));
    if (crossingCount == (uint)(1u)) {
      if (exactIndex == ((ulong)(-1))) {
      } else {
        ulong exactFlat = (pointBegin + exactIndex);
        exactPoint = (uint)(1u);
        lowerPoint = exactIndex;
        elevation = verticalCoordinates[exactFlat];
      }
    }
    elevationOut[gid] = elevation;
    lowerPointOut[gid] = lowerPoint;
    crossingCountOut[gid] = crossingCount;
    exactPointOut[gid] = exactPoint;
    crossingValidOut[gid] = crossingValid;
  }
}

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64(__global const double* alphaCell, __global const double* alphaMinimum, __global const double* alphaMaximum, __global const ulong* incidentOffsets, __global const double* gradientDotDisplacements, __global const uint* nCells, __global double* psiOut, __global ulong* limitingFaceOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint cellCount = nCells[0];
  if (gid < cellCount) {
    uint nextCell = (gid + (uint)(1u));
    ulong incidentBegin = incidentOffsets[gid];
    ulong incidentEnd = incidentOffsets[nextCell];
    ulong incidentCount = (incidentEnd - incidentBegin);
    double alpha = alphaCell[gid];
    double alphaMin = alphaMinimum[gid];
    double alphaMax = alphaMaximum[gid];
    double psi = as_double(0x3ff0000000000000ul);
    ulong limitingFace = ((ulong)(-1));
    for (int face = 0; face < (int)(incidentCount); ++face) {
      ulong flatIndex = (incidentBegin + ((ulong)(face)));
      double delta = gradientDotDisplacements[flatIndex];
      double rawCandidate = (((as_double(0x0000000000000000ul) < delta)) ? ((((((alphaMax - alpha) / delta) < as_double(0x3ff0000000000000ul))) ? (((alphaMax - alpha) / delta)) : (as_double(0x3ff0000000000000ul)))) : ((((delta < as_double(0x0000000000000000ul))) ? ((((((alphaMin - alpha) / delta) < as_double(0x3ff0000000000000ul))) ? (((alphaMin - alpha) / delta)) : (as_double(0x3ff0000000000000ul)))) : (as_double(0x3ff0000000000000ul)))));
      double lowerClamped = (((rawCandidate < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (rawCandidate));
      double candidate = (((as_double(0x3ff0000000000000ul) < lowerClamped)) ? (as_double(0x3ff0000000000000ul)) : (lowerClamped));
      if (candidate < psi) {
        psi = candidate;
        limitingFace = ((ulong)(face));
      }
    }
    psiOut[gid] = psi;
    limitingFaceOut[gid] = limitingFace;
  }
}

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64(__global const double* velocityGradient00, __global const double* velocityGradient01, __global const double* velocityGradient02, __global const double* velocityGradient10, __global const double* velocityGradient11, __global const double* velocityGradient12, __global const double* velocityGradient20, __global const double* velocityGradient21, __global const double* velocityGradient22, __global const double* dynamicViscosity, __global const double* fluidAreaVectorX, __global const double* fluidAreaVectorY, __global const double* fluidAreaVectorZ, __global const double* faceCentroidX, __global const double* faceCentroidY, __global const double* faceCentroidZ, __global const double* hullLaminarDatum, __global const uint* nFaces, __global double* areaOut, __global double* forceXOut, __global double* forceYOut, __global double* forceZOut, __global double* momentXOut, __global double* momentYOut, __global double* momentZOut) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double gradient00 = velocityGradient00[gid];
    double gradient01 = velocityGradient01[gid];
    double gradient02 = velocityGradient02[gid];
    double gradient10 = velocityGradient10[gid];
    double gradient11 = velocityGradient11[gid];
    double gradient12 = velocityGradient12[gid];
    double gradient20 = velocityGradient20[gid];
    double gradient21 = velocityGradient21[gid];
    double gradient22 = velocityGradient22[gid];
    double trace = ((gradient00 + gradient11) + gradient22);
    double traceThird = (trace / as_double(0x4008000000000000ul));
    double twoMu = (as_double(0x4000000000000000ul) * dynamicViscosity[gid]);
    double strain00 = ((as_double(0x3fe0000000000000ul) * (gradient00 + gradient00)) - traceThird);
    double strain01 = (as_double(0x3fe0000000000000ul) * (gradient01 + gradient10));
    double strain02 = (as_double(0x3fe0000000000000ul) * (gradient02 + gradient20));
    double strain10 = (as_double(0x3fe0000000000000ul) * (gradient10 + gradient01));
    double strain11 = ((as_double(0x3fe0000000000000ul) * (gradient11 + gradient11)) - traceThird);
    double strain12 = (as_double(0x3fe0000000000000ul) * (gradient12 + gradient21));
    double strain20 = (as_double(0x3fe0000000000000ul) * (gradient20 + gradient02));
    double strain21 = (as_double(0x3fe0000000000000ul) * (gradient21 + gradient12));
    double strain22 = ((as_double(0x3fe0000000000000ul) * (gradient22 + gradient22)) - traceThird);
    double tau00 = (twoMu * strain00);
    double tau01 = (twoMu * strain01);
    double tau02 = (twoMu * strain02);
    double tau10 = (twoMu * strain10);
    double tau11 = (twoMu * strain11);
    double tau12 = (twoMu * strain12);
    double tau20 = (twoMu * strain20);
    double tau21 = (twoMu * strain21);
    double tau22 = (twoMu * strain22);
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double minusAreaX = (areaX * as_double(0xbff0000000000000ul));
    double minusAreaY = (areaY * as_double(0xbff0000000000000ul));
    double minusAreaZ = (areaZ * as_double(0xbff0000000000000ul));
    double forceX = (((tau00 * minusAreaX) + (tau01 * minusAreaY)) + (tau02 * minusAreaZ));
    double forceY = (((tau10 * minusAreaX) + (tau11 * minusAreaY)) + (tau12 * minusAreaZ));
    double forceZ = (((tau20 * minusAreaX) + (tau21 * minusAreaY)) + (tau22 * minusAreaZ));
    double momentArmX = (faceCentroidX[gid] - hullLaminarDatum[0]);
    double momentArmY = (faceCentroidY[gid] - hullLaminarDatum[1]);
    double momentArmZ = (faceCentroidZ[gid] - hullLaminarDatum[2]);
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

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64 = R"kernel(
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

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64(__global const double* fluidAreaVectorX, __global const double* fluidAreaVectorY, __global const double* fluidAreaVectorZ, __global const double* faceCentroidX, __global const double* faceCentroidY, __global const double* faceCentroidZ, __global const double* ownerCentroidX, __global const double* ownerCentroidY, __global const double* ownerCentroidZ, __global const double* pressureGradientOwnerX, __global const double* pressureGradientOwnerY, __global const double* pressureGradientOwnerZ, __global const double* pRghOwner, __global const double* densityOwner, __global const double* hullPressureDatum, __global const uint* nFaces, __global double* areaOut, __global double* pRghFaceOut, __global double* absolutePressureOut, __global double* relativePressureOut, __global double* forceXOut, __global double* forceYOut, __global double* forceZOut, __global double* momentXOut, __global double* momentYOut, __global double* momentZOut) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double inverseArea = (as_double(0x3ff0000000000000ul) / area);
    double normalX = (areaX * inverseArea);
    double normalY = (areaY * inverseArea);
    double normalZ = (areaZ * inverseArea);
    double normalGradient = (((pressureGradientOwnerX[gid] * normalX) + (pressureGradientOwnerY[gid] * normalY)) + (pressureGradientOwnerZ[gid] * normalZ));
    double tangentGradientX = (pressureGradientOwnerX[gid] - (normalX * normalGradient));
    double tangentGradientY = (pressureGradientOwnerY[gid] - (normalY * normalGradient));
    double tangentGradientZ = (pressureGradientOwnerZ[gid] - (normalZ * normalGradient));
    double ownerToFaceX = (faceCentroidX[gid] - ownerCentroidX[gid]);
    double ownerToFaceY = (faceCentroidY[gid] - ownerCentroidY[gid]);
    double ownerToFaceZ = (faceCentroidZ[gid] - ownerCentroidZ[gid]);
    double pRghFace = (pRghOwner[gid] + (((tangentGradientX * ownerToFaceX) + (tangentGradientY * ownerToFaceY)) + (tangentGradientZ * ownerToFaceZ)));
    double referenceToFaceX = (faceCentroidX[gid] - hullPressureDatum[3]);
    double referenceToFaceY = (faceCentroidY[gid] - hullPressureDatum[4]);
    double referenceToFaceZ = (faceCentroidZ[gid] - hullPressureDatum[5]);
    double head = (densityOwner[gid] * (((hullPressureDatum[0] * referenceToFaceX) + (hullPressureDatum[1] * referenceToFaceY)) + (hullPressureDatum[2] * referenceToFaceZ)));
    double absolutePressure = (pRghFace + head);
    double relativePressure = ((pRghFace - hullPressureDatum[9]) + head);
    double forceX = (areaX * relativePressure);
    double forceY = (areaY * relativePressure);
    double forceZ = (areaZ * relativePressure);
    double momentArmX = (faceCentroidX[gid] - hullPressureDatum[6]);
    double momentArmY = (faceCentroidY[gid] - hullPressureDatum[7]);
    double momentArmZ = (faceCentroidZ[gid] - hullPressureDatum[8]);
    double momentX = ((momentArmY * forceZ) - (momentArmZ * forceY));
    double momentY = ((momentArmZ * forceX) - (momentArmX * forceZ));
    double momentZ = ((momentArmX * forceY) - (momentArmY * forceX));
    areaOut[gid] = area;
    pRghFaceOut[gid] = pRghFace;
    absolutePressureOut[gid] = absolutePressure;
    relativePressureOut[gid] = relativePressure;
    forceXOut[gid] = forceX;
    forceYOut[gid] = forceY;
    forceZOut[gid] = forceZ;
    momentXOut[gid] = momentX;
    momentYOut[gid] = momentY;
    momentZOut[gid] = momentZ;
  }
}

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64 = R"kernel(
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

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64(__global const double* times, __global const double* values, __global const uint* count, __global double* scratchA, __global double* scratchB, __global double* firstTimeOut, __global double* lastTimeOut, __global double* meanOut, __global double* sampleStandardDeviationOut, __global double* minimumOut, __global double* maximumOut, __global double* driftSlopeOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint sampleCount = count[0];
  if (gid == (uint)(0u)) {
    double firstTime = times[(uint)(0u)];
    uint lastIndex = (sampleCount - (uint)(1u));
    double lastTime = times[lastIndex];
    double minimum = values[(uint)(0u)];
    double maximum = values[(uint)(0u)];
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double sampleValue = values[sampleIndex];
      scratchA[sampleIndex] = sampleValue;
      if (sampleValue < minimum) {
        minimum = sampleValue;
      }
      if (maximum < sampleValue) {
        maximum = sampleValue;
      }
    }
    uint valueActive = sampleCount;
    uint valueCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < valueActive) {
        uint pairs = (valueActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((valueCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((valueCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (valueCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (valueActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (valueActive - (uint)(1u));
          double carryValue = (((valueCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (valueCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        valueActive = (pairs + odd);
        valueCurrentA = (((valueCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double valueSum = (((valueCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double sampleCountF64 = ((double)(sampleCount));
    double mean = (valueSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      scratchA[sampleIndex] = times[sampleIndex];
    }
    uint timeActive = sampleCount;
    uint timeCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < timeActive) {
        uint pairs = (timeActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((timeCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (timeActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (timeActive - (uint)(1u));
          double carryValue = (((timeCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeActive = (pairs + odd);
        timeCurrentA = (((timeCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double timeSum = (((timeCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double meanTime = (timeSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double deviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (deviation * deviation);
    }
    uint varianceActive = sampleCount;
    uint varianceCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < varianceActive) {
        uint pairs = (varianceActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((varianceCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((varianceCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (varianceCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (varianceActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (varianceActive - (uint)(1u));
          double carryValue = (((varianceCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (varianceCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        varianceActive = (pairs + odd);
        varianceCurrentA = (((varianceCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double varianceSum = (((varianceCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double sampleDenominator = ((double)((sampleCount - (uint)(1u))));
    double sampleStandardDeviation = sqrt((varianceSum / sampleDenominator));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      double valueDeviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (timeDeviation * valueDeviation);
    }
    uint covarianceActive = sampleCount;
    uint covarianceCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < covarianceActive) {
        uint pairs = (covarianceActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (covarianceCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (covarianceActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (covarianceActive - (uint)(1u));
          double carryValue = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (covarianceCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        covarianceActive = (pairs + odd);
        covarianceCurrentA = (((covarianceCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double covarianceSum = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      scratchA[sampleIndex] = (timeDeviation * timeDeviation);
    }
    uint timeVarianceActive = sampleCount;
    uint timeVarianceCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < timeVarianceActive) {
        uint pairs = (timeVarianceActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeVarianceCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (timeVarianceActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (timeVarianceActive - (uint)(1u));
          double carryValue = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeVarianceCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeVarianceActive = (pairs + odd);
        timeVarianceCurrentA = (((timeVarianceCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double timeVarianceSum = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double driftSlope = (covarianceSum / timeVarianceSum);
    firstTimeOut[(uint)(0u)] = firstTime;
    lastTimeOut[(uint)(0u)] = lastTime;
    meanOut[(uint)(0u)] = mean;
    sampleStandardDeviationOut[(uint)(0u)] = sampleStandardDeviation;
    minimumOut[(uint)(0u)] = minimum;
    maximumOut[(uint)(0u)] = maximum;
    driftSlopeOut[(uint)(0u)] = driftSlope;
  }
}

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64(__global const double* volumeFlux, __global const double* faceArea, __global const double* maximumIncidentFaceSpeed, __global const double* interfaceNormalDotArea, __global const double* alphaFace, __global const double* compressionDatum, __global const uint* nFaces, __global double* compressionSpeedOut, __global double* compressionFluxOut, __global double* alphaCompressionFluxOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint faceCount = nFaces[0];
  if (gid < faceCount) {
    double coefficient = compressionDatum[0];
    double alpha = alphaFace[gid];
    double compressionSpeed = ((((coefficient * maximumIncidentFaceSpeed[gid]) < ((coefficient * ((((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid]))) ? ((coefficient * maximumIncidentFaceSpeed[gid])) : (((coefficient * ((((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid])));
    double compressionFlux = (compressionSpeed * interfaceNormalDotArea[gid]);
    double alphaCompressionFlux = ((compressionFlux * alpha) * (as_double(0x3ff0000000000000ul) - alpha));
    compressionSpeedOut[gid] = compressionSpeed;
    compressionFluxOut[gid] = compressionFlux;
    alphaCompressionFluxOut[gid] = alphaCompressionFlux;
  }
}

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64 = R"kernel(
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

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64 = R"kernel(
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

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64(__global const double* alpha1, __global const double* k, __global const double* omega, __global const double* wallDistance, __global const double* gradientDotProduct, __global const double* strainMagnitudeSquared2, __global const double* datum, __global const uint* nCells, __global double* densityOut, __global double* dynamicViscosityOut, __global double* kinematicViscosityOut, __global double* crossDiffusionOut, __global double* positiveCrossDiffusionOut, __global double* argument1Out, __global double* argument2Out, __global double* f1Out, __global double* f2Out, __global double* turbulentKinematicViscosityOut, __global double* turbulentDynamicViscosityOut, __global double* effectiveDynamicViscosityOut, __global double* nutToNuOut, __global double* gammaOut, __global double* betaOut, __global double* sigmaKOut, __global double* sigmaOmegaOut, __global double* productionRawOut, __global double* productionLimitedOut, __global double* kExplicitSourceOut, __global double* kImplicitSinkOut, __global double* omegaBaseSourceOut, __global double* crossLagOut, __global double* omegaCrossExplicitSourceOut, __global double* omegaCrossImplicitSinkOut, __global double* omegaImplicitSinkOut, __global double* kEffectiveDiffusivityOut, __global double* omegaEffectiveDiffusivityOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint cellCount = nCells[0];
  if (gid < cellCount) {
    double alpha = alpha1[gid];
    double kValue = k[gid];
    double omegaValue = omega[gid];
    double distance = wallDistance[gid];
    double gradientDot = gradientDotProduct[gid];
    double strain2 = strainMagnitudeSquared2[gid];
    double oneMinusAlpha = (as_double(0x3ff0000000000000ul) - alpha);
    double density = ((alpha * datum[0]) + (oneMinusAlpha * datum[1]));
    double dynamicViscosity = ((alpha * datum[2]) + (oneMinusAlpha * datum[3]));
    double kinematicViscosity = (dynamicViscosity / density);
    double crossDiffusion = (((as_double(0x4000000000000000ul) * as_double(0x3feb645a1cac0831ul)) / omegaValue) * gradientDot);
    double positiveCrossDiffusion = (((crossDiffusion < as_double(0x3ddb7cdfd9d7bdbbul))) ? (as_double(0x3ddb7cdfd9d7bdbbul)) : (crossDiffusion));
    double rootK = sqrt(kValue);
    double distanceSquared = (distance * distance);
    double firstA = (rootK / ((as_double(0x3fb70a3d70a3d70aul) * omegaValue) * distance));
    double firstB = ((as_double(0x407f400000000000ul) * kinematicViscosity) / (distanceSquared * omegaValue));
    double firstC = (((as_double(0x4010000000000000ul) * as_double(0x3feb645a1cac0831ul)) * kValue) / (positiveCrossDiffusion * distanceSquared));
    double argument1 = (((firstC < (((firstA < firstB)) ? (firstB) : (firstA)))) ? (firstC) : ((((firstA < firstB)) ? (firstB) : (firstA))));
    double argument2 = ((((as_double(0x4000000000000000ul) * firstA) < firstB)) ? (firstB) : ((as_double(0x4000000000000000ul) * firstA)));
    double f1Value = (((((argument1 * argument1) * (argument1 * argument1)) < as_double(0x4034000000000000ul))) ? (tanh(((argument1 * argument1) * (argument1 * argument1)))) : (as_double(0x3ff0000000000000ul)));
    double f2Value = ((((argument2 * argument2) < as_double(0x4034000000000000ul))) ? (tanh((argument2 * argument2))) : (as_double(0x3ff0000000000000ul)));
    double strainMagnitude = sqrt(strain2);
    double nutDenominator = ((((as_double(0x3fd3d70a3d70a3d7ul) * omegaValue) < ((as_double(0x3ff0000000000000ul) * f2Value) * strainMagnitude))) ? (((as_double(0x3ff0000000000000ul) * f2Value) * strainMagnitude)) : ((as_double(0x3fd3d70a3d70a3d7ul) * omegaValue)));
    double nut = ((as_double(0x3fd3d70a3d70a3d7ul) * kValue) / nutDenominator);
    double turbulentDynamicViscosity = (density * nut);
    double effectiveDynamicViscosity = (dynamicViscosity + turbulentDynamicViscosity);
    double nutToNu = (nut / (dynamicViscosity / density));
    double oneMinusF1 = (as_double(0x3ff0000000000000ul) - f1Value);
    double gamma = ((f1Value * as_double(0x3fe1c71c71c71c72ul)) + (oneMinusF1 * as_double(0x3fdc28f5c28f5c29ul)));
    double beta = ((f1Value * as_double(0x3fb3333333333333ul)) + (oneMinusF1 * as_double(0x3fb532617c1bda51ul)));
    double sigmaK = ((f1Value * as_double(0x3feb333333333333ul)) + (oneMinusF1 * as_double(0x3ff0000000000000ul)));
    double sigmaOmega = ((f1Value * as_double(0x3fe0000000000000ul)) + (oneMinusF1 * as_double(0x3feb645a1cac0831ul)));
    double productionRaw = (turbulentDynamicViscosity * strain2);
    double productionLimit = ((((as_double(0x4024000000000000ul) * as_double(0x3fb70a3d70a3d70aul)) * density) * kValue) * omegaValue);
    double productionLimited = (((productionLimit < productionRaw)) ? (productionLimit) : (productionRaw));
    double kExplicitSource = (((productionLimited < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (productionLimited));
    double kImplicitSink = ((as_double(0x3fb70a3d70a3d70aul) * density) * omegaValue);
    double omegaBaseSource = ((gamma * density) * strain2);
    double admittedOmega = (((omegaValue < as_double(0x3d719799812dea11ul))) ? (as_double(0x3d719799812dea11ul)) : (omegaValue));
    double crossLag = (((((as_double(0x4000000000000000ul) * oneMinusF1) * density) * as_double(0x3feb645a1cac0831ul)) / admittedOmega) * gradientDot);
    double omegaCrossExplicitSource = (((crossLag < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (crossLag));
    double omegaCrossImplicitSink = (((((-crossLag) < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((-crossLag))) / admittedOmega);
    double omegaImplicitSink = (((beta * density) * omegaValue) + omegaCrossImplicitSink);
    double kEffectiveDiffusivity = (dynamicViscosity + (sigmaK * turbulentDynamicViscosity));
    double omegaEffectiveDiffusivity = (dynamicViscosity + (sigmaOmega * turbulentDynamicViscosity));
    densityOut[gid] = density;
    dynamicViscosityOut[gid] = dynamicViscosity;
    kinematicViscosityOut[gid] = kinematicViscosity;
    crossDiffusionOut[gid] = crossDiffusion;
    positiveCrossDiffusionOut[gid] = positiveCrossDiffusion;
    argument1Out[gid] = argument1;
    argument2Out[gid] = argument2;
    f1Out[gid] = f1Value;
    f2Out[gid] = f2Value;
    turbulentKinematicViscosityOut[gid] = nut;
    turbulentDynamicViscosityOut[gid] = turbulentDynamicViscosity;
    effectiveDynamicViscosityOut[gid] = effectiveDynamicViscosity;
    nutToNuOut[gid] = nutToNu;
    gammaOut[gid] = gamma;
    betaOut[gid] = beta;
    sigmaKOut[gid] = sigmaK;
    sigmaOmegaOut[gid] = sigmaOmega;
    productionRawOut[gid] = productionRaw;
    productionLimitedOut[gid] = productionLimited;
    kExplicitSourceOut[gid] = kExplicitSource;
    kImplicitSinkOut[gid] = kImplicitSink;
    omegaBaseSourceOut[gid] = omegaBaseSource;
    crossLagOut[gid] = crossLag;
    omegaCrossExplicitSourceOut[gid] = omegaCrossExplicitSource;
    omegaCrossImplicitSinkOut[gid] = omegaCrossImplicitSink;
    omegaImplicitSinkOut[gid] = omegaImplicitSink;
    kEffectiveDiffusivityOut[gid] = kEffectiveDiffusivity;
    omegaEffectiveDiffusivityOut[gid] = omegaEffectiveDiffusivity;
  }
}

)kernel";
const char* k_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64 = R"kernel(
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

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64[] = {
  { "probeOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "verticalCoordinates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaValues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nProbes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "elevationOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lowerPointOut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "crossingCountOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "exactPointOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "crossingValidOut", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  9,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64[] = {
  { "alphaCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaMinimum", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaMaximum", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "incidentOffsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradientDotDisplacements", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "psiOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "limitingFaceOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  8,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64[] = {
  { "velocityGradient00", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient01", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient02", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient10", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient11", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient12", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient20", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient21", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "velocityGradient22", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dynamicViscosity", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorX", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorY", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorZ", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidX", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidY", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidZ", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hullLaminarDatum", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 24, 24, 256, nullptr, 0, 0 },
  { "nFaces", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "areaOut", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceXOut", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceYOut", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceZOut", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentXOut", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentYOut", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentZOut", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  25,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64[] = {
  { "pressureForceXIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureForceYIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureForceZIn", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousForceXIn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousForceYIn", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousForceZIn", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureMomentXIn", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureMomentYIn", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureMomentZIn", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousMomentXIn", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousMomentYIn", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousMomentZIn", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "activeCount", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pressureForceXOut", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureForceYOut", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureForceZOut", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousForceXOut", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousForceYOut", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousForceZOut", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureMomentXOut", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureMomentYOut", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureMomentZOut", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousMomentXOut", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousMomentYOut", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "viscousMomentZOut", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  25,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64[] = {
  { "fluidAreaVectorX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerCentroidX", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerCentroidY", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerCentroidZ", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureGradientOwnerX", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureGradientOwnerY", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pressureGradientOwnerZ", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pRghOwner", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "densityOwner", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hullPressureDatum", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 80, 80, 256, nullptr, 0, 0 },
  { "nFaces", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "areaOut", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pRghFaceOut", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "absolutePressureOut", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "relativePressureOut", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceXOut", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceYOut", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceZOut", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentXOut", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentYOut", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentZOut", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  26,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64[] = {
  { "wallShearOnFluidX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallShearOnFluidY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallShearOnFluidZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fluidAreaVectorZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidX", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidY", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCentroidZ", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hullSstDatum", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 24, 24, 256, nullptr, 0, 0 },
  { "nFaces", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "areaOut", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceXOut", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceYOut", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forceZOut", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentXOut", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentYOut", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentZOut", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  18,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64[] = {
  { "times", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scratchA", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scratchB", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "firstTimeOut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lastTimeOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "meanOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sampleStandardDeviationOut", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minimumOut", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maximumOut", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "driftSlopeOut", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  12,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64[] = {
  { "volumeFlux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceArea", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maximumIncidentFaceSpeed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "interfaceNormalDotArea", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaFace", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "compressionDatum", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "nFaces", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "compressionSpeedOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "compressionFluxOut", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaCompressionFluxOut", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  10,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64[] = {
  { "density", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "volume", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "streamwiseCoordinate", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "acceptedVelocityX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "acceptedVelocityY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "acceptedVelocityZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dampingDatum", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 56, 56, 256, nullptr, 0, 0 },
  { "nCells", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "lambdaOut", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "implicitDiagonalOut", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "implicitTargetRhsXOut", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "implicitTargetRhsYOut", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "implicitTargetRhsZOut", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentumSourceXOut", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentumSourceYOut", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "momentumSourceZOut", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signedPowerOut", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signedWorkOut", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "activeOut", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  19,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "liquidFraction", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "datum", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 96, 96, 256, nullptr, 0, 0 },
  { "nCells", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "densityOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pRghOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "absolutePressureOut", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  9,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64[] = {
  { "alpha1", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallDistance", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradientDotProduct", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strainMagnitudeSquared2", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "datum", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 32, 32, 256, nullptr, 0, 0 },
  { "nCells", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "densityOut", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dynamicViscosityOut", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kinematicViscosityOut", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "crossDiffusionOut", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positiveCrossDiffusionOut", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "argument1Out", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "argument2Out", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "f1Out", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "f2Out", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "turbulentKinematicViscosityOut", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "turbulentDynamicViscosityOut", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "effectiveDynamicViscosityOut", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nutToNuOut", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gammaOut", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "betaOut", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigmaKOut", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigmaOmegaOut", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "productionRawOut", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "productionLimitedOut", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kExplicitSourceOut", 27, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kImplicitSinkOut", 28, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaBaseSourceOut", 29, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "crossLagOut", 30, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaCrossExplicitSourceOut", 31, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaCrossImplicitSinkOut", 32, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaImplicitSinkOut", 33, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kEffectiveDiffusivityOut", 34, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaEffectiveDiffusivityOut", 35, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64 = {
  1,
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  36,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64
};

const KernelArgumentInfo kAbiArgs_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64[] = {
  { "tangentialVelocityX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tangentialVelocityY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tangentialVelocityZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallDistance", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kinematicViscosity", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "density", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dynamicViscosity", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "referenceVelocity", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFaces", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "densityOut", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tangentMagnitudeOut", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "frictionVelocityOut", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tangentUnitXOut", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tangentUnitYOut", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tangentUnitZOut", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallShearOnFluidXOut", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallShearOnFluidYOut", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallShearOnFluidZOut", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uPlusOut", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "yPlusOut", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wallShearMagnitudeOut", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaViscousOut", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaLogarithmicOut", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaWallOut", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "turbulentKinematicViscosityWallOut", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "finalResidualOut", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "iterationsOut", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "zeroVelocityBranchOut", 27, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64 = {
  1,
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "opencl",
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "kernel:opencl:Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "abi-r1:opencl:Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  28,
  kAbiArgs_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64
};

bool tryGetKernelAbiManifest_opencl_unmapped(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64;
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64") {
    out = &kAbiManifest_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_unmapped(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64";
    std::string_view sv(k_opencl_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

