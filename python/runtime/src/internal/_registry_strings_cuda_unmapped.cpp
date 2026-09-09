// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_unmapped` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64(const unsigned long long* probeOffsets, const double* verticalCoordinates, const double* alphaValues, const unsigned int* nProbes, double* elevationOut, unsigned long long* lowerPointOut, unsigned int* crossingCountOut, unsigned int* exactPointOut, unsigned int* crossingValidOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int probeCount = nProbes[0];
  if (gid < probeCount) {
    unsigned int nextProbe = (gid + 1u);
    unsigned long long pointBegin = probeOffsets[gid];
    unsigned long long pointEnd = probeOffsets[nextProbe];
    unsigned long long pointCount = (pointEnd - pointBegin);
    unsigned long long segmentCount = (pointCount - ((unsigned long long)(1)));
    unsigned int crossingCount = 0u;
    unsigned long long exactIndex = ((unsigned long long)(-1));
    double elevation = __longlong_as_double(0x0000000000000000ull);
    unsigned long long lowerPoint = ((unsigned long long)(-1));
    unsigned int exactPoint = 0u;
    for (int point = 0; point < (int)(pointCount); ++point) {
      unsigned long long pointLocal = ((unsigned long long)(point));
      unsigned long long pointFlat = (pointBegin + pointLocal);
      double pointAlpha = alphaValues[pointFlat];
      if (pointAlpha == __longlong_as_double(0x3fe0000000000000ull)) {
        unsigned int exactCrossingCount = (crossingCount + 1u);
        crossingCount = exactCrossingCount;
        exactIndex = pointLocal;
      }
    }
    for (int segment = 0; segment < (int)(segmentCount); ++segment) {
      unsigned long long segmentLocal = ((unsigned long long)(segment));
      unsigned long long segmentFlat = (pointBegin + segmentLocal);
      unsigned long long nextSegmentFlat = (segmentFlat + ((unsigned long long)(1)));
      double coordinateA = verticalCoordinates[segmentFlat];
      double coordinateB = verticalCoordinates[nextSegmentFlat];
      double a = (alphaValues[segmentFlat] - __longlong_as_double(0x3fe0000000000000ull));
      double b = (alphaValues[nextSegmentFlat] - __longlong_as_double(0x3fe0000000000000ull));
      if (a < __longlong_as_double(0x0000000000000000ull)) {
        if (__longlong_as_double(0x0000000000000000ull) < b) {
          unsigned int strictCrossingCount = (crossingCount + 1u);
          crossingCount = strictCrossingCount;
          lowerPoint = segmentLocal;
          if (strictCrossingCount == 1u) {
            elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
          }
        }
      } else {
        if (__longlong_as_double(0x0000000000000000ull) < a) {
          if (b < __longlong_as_double(0x0000000000000000ull)) {
            unsigned int strictCrossingCount = (crossingCount + 1u);
            crossingCount = strictCrossingCount;
            lowerPoint = segmentLocal;
            if (strictCrossingCount == 1u) {
              elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
            }
          }
        }
      }
    }
    unsigned int crossingValid = (((crossingCount == 1u)) ? (1u) : (0u));
    if (crossingCount == 1u) {
      if (exactIndex == ((unsigned long long)(-1))) {
      } else {
        unsigned long long exactFlat = (pointBegin + exactIndex);
        exactPoint = 1u;
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
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64(const double* alphaCell, const double* alphaMinimum, const double* alphaMaximum, const unsigned long long* incidentOffsets, const double* gradientDotDisplacements, const unsigned int* nCells, double* psiOut, unsigned long long* limitingFaceOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int cellCount = nCells[0];
  if (gid < cellCount) {
    unsigned int nextCell = (gid + 1u);
    unsigned long long incidentBegin = incidentOffsets[gid];
    unsigned long long incidentEnd = incidentOffsets[nextCell];
    unsigned long long incidentCount = (incidentEnd - incidentBegin);
    double alpha = alphaCell[gid];
    double alphaMin = alphaMinimum[gid];
    double alphaMax = alphaMaximum[gid];
    double psi = __longlong_as_double(0x3ff0000000000000ull);
    unsigned long long limitingFace = ((unsigned long long)(-1));
    for (int face = 0; face < (int)(incidentCount); ++face) {
      unsigned long long flatIndex = (incidentBegin + ((unsigned long long)(face)));
      double delta = gradientDotDisplacements[flatIndex];
      double rawCandidate = (((__longlong_as_double(0x0000000000000000ull) < delta)) ? ((((((alphaMax - alpha) / delta) < __longlong_as_double(0x3ff0000000000000ull))) ? (((alphaMax - alpha) / delta)) : (__longlong_as_double(0x3ff0000000000000ull)))) : ((((delta < __longlong_as_double(0x0000000000000000ull))) ? ((((((alphaMin - alpha) / delta) < __longlong_as_double(0x3ff0000000000000ull))) ? (((alphaMin - alpha) / delta)) : (__longlong_as_double(0x3ff0000000000000ull)))) : (__longlong_as_double(0x3ff0000000000000ull)))));
      double lowerClamped = (((rawCandidate < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (rawCandidate));
      double candidate = (((__longlong_as_double(0x3ff0000000000000ull) < lowerClamped)) ? (__longlong_as_double(0x3ff0000000000000ull)) : (lowerClamped));
      if (candidate < psi) {
        psi = candidate;
        limitingFace = ((unsigned long long)(face));
      }
    }
    psiOut[gid] = psi;
    limitingFaceOut[gid] = limitingFace;
  }
}

)kernel";
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64(const double* velocityGradient00, const double* velocityGradient01, const double* velocityGradient02, const double* velocityGradient10, const double* velocityGradient11, const double* velocityGradient12, const double* velocityGradient20, const double* velocityGradient21, const double* velocityGradient22, const double* dynamicViscosity, const double* fluidAreaVectorX, const double* fluidAreaVectorY, const double* fluidAreaVectorZ, const double* faceCentroidX, const double* faceCentroidY, const double* faceCentroidZ, const double* hullLaminarDatum, const unsigned int* nFaces, double* areaOut, double* forceXOut, double* forceYOut, double* forceZOut, double* momentXOut, double* momentYOut, double* momentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
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
    double traceThird = (trace / __longlong_as_double(0x4008000000000000ull));
    double twoMu = (__longlong_as_double(0x4000000000000000ull) * dynamicViscosity[gid]);
    double strain00 = ((__longlong_as_double(0x3fe0000000000000ull) * (gradient00 + gradient00)) - traceThird);
    double strain01 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient01 + gradient10));
    double strain02 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient02 + gradient20));
    double strain10 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient10 + gradient01));
    double strain11 = ((__longlong_as_double(0x3fe0000000000000ull) * (gradient11 + gradient11)) - traceThird);
    double strain12 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient12 + gradient21));
    double strain20 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient20 + gradient02));
    double strain21 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient21 + gradient12));
    double strain22 = ((__longlong_as_double(0x3fe0000000000000ull) * (gradient22 + gradient22)) - traceThird);
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
    double minusAreaX = (areaX * __longlong_as_double(0xbff0000000000000ull));
    double minusAreaY = (areaY * __longlong_as_double(0xbff0000000000000ull));
    double minusAreaZ = (areaZ * __longlong_as_double(0xbff0000000000000ull));
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
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64(const double* pressureForceXIn, const double* pressureForceYIn, const double* pressureForceZIn, const double* viscousForceXIn, const double* viscousForceYIn, const double* viscousForceZIn, const double* pressureMomentXIn, const double* pressureMomentYIn, const double* pressureMomentZIn, const double* viscousMomentXIn, const double* viscousMomentYIn, const double* viscousMomentZIn, const unsigned int* activeCount, double* pressureForceXOut, double* pressureForceYOut, double* pressureForceZOut, double* viscousForceXOut, double* viscousForceYOut, double* viscousForceZOut, double* pressureMomentXOut, double* pressureMomentYOut, double* pressureMomentZOut, double* viscousMomentXOut, double* viscousMomentYOut, double* viscousMomentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int _active = activeCount[0];
  unsigned int activeHalf = (_active / 2u);
  unsigned int outputCount = (activeHalf + (_active % 2u));
  if (gid < outputCount) {
    unsigned int leftIndex = (gid * 2u);
    unsigned int rightIndex = (leftIndex + 1u);
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
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64(const double* fluidAreaVectorX, const double* fluidAreaVectorY, const double* fluidAreaVectorZ, const double* faceCentroidX, const double* faceCentroidY, const double* faceCentroidZ, const double* ownerCentroidX, const double* ownerCentroidY, const double* ownerCentroidZ, const double* pressureGradientOwnerX, const double* pressureGradientOwnerY, const double* pressureGradientOwnerZ, const double* pRghOwner, const double* densityOwner, const double* hullPressureDatum, const unsigned int* nFaces, double* areaOut, double* pRghFaceOut, double* absolutePressureOut, double* relativePressureOut, double* forceXOut, double* forceYOut, double* forceZOut, double* momentXOut, double* momentYOut, double* momentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double inverseArea = (__longlong_as_double(0x3ff0000000000000ull) / area);
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
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64(const double* wallShearOnFluidX, const double* wallShearOnFluidY, const double* wallShearOnFluidZ, const double* fluidAreaVectorX, const double* fluidAreaVectorY, const double* fluidAreaVectorZ, const double* faceCentroidX, const double* faceCentroidY, const double* faceCentroidZ, const double* hullSstDatum, const unsigned int* nFaces, double* areaOut, double* forceXOut, double* forceYOut, double* forceZOut, double* momentXOut, double* momentYOut, double* momentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
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
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64(const double* times, const double* values, const unsigned int* count, double* scratchA, double* scratchB, double* firstTimeOut, double* lastTimeOut, double* meanOut, double* sampleStandardDeviationOut, double* minimumOut, double* maximumOut, double* driftSlopeOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int sampleCount = count[0];
  if (gid == 0u) {
    double firstTime = times[0u];
    unsigned int lastIndex = (sampleCount - 1u);
    double lastTime = times[lastIndex];
    double minimum = values[0u];
    double maximum = values[0u];
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double sampleValue = values[sampleIndex];
      scratchA[sampleIndex] = sampleValue;
      if (sampleValue < minimum) {
        minimum = sampleValue;
      }
      if (maximum < sampleValue) {
        maximum = sampleValue;
      }
    }
    unsigned int valueActive = sampleCount;
    unsigned int valueCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < valueActive) {
        unsigned int pairs = (valueActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((valueCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((valueCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (valueCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (valueActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (valueActive - 1u);
          double carryValue = (((valueCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (valueCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        valueActive = (pairs + odd);
        valueCurrentA = (((valueCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double valueSum = (((valueCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double sampleCountF64 = ((double)(sampleCount));
    double mean = (valueSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      scratchA[sampleIndex] = times[sampleIndex];
    }
    unsigned int timeActive = sampleCount;
    unsigned int timeCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < timeActive) {
        unsigned int pairs = (timeActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((timeCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (timeActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (timeActive - 1u);
          double carryValue = (((timeCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeActive = (pairs + odd);
        timeCurrentA = (((timeCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double timeSum = (((timeCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double meanTime = (timeSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double deviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (deviation * deviation);
    }
    unsigned int varianceActive = sampleCount;
    unsigned int varianceCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < varianceActive) {
        unsigned int pairs = (varianceActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((varianceCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((varianceCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (varianceCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (varianceActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (varianceActive - 1u);
          double carryValue = (((varianceCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (varianceCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        varianceActive = (pairs + odd);
        varianceCurrentA = (((varianceCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double varianceSum = (((varianceCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double sampleDenominator = ((double)((sampleCount - 1u)));
    double sampleStandardDeviation = sqrt((varianceSum / sampleDenominator));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      double valueDeviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (timeDeviation * valueDeviation);
    }
    unsigned int covarianceActive = sampleCount;
    unsigned int covarianceCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < covarianceActive) {
        unsigned int pairs = (covarianceActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((covarianceCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((covarianceCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (covarianceCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (covarianceActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (covarianceActive - 1u);
          double carryValue = (((covarianceCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (covarianceCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        covarianceActive = (pairs + odd);
        covarianceCurrentA = (((covarianceCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double covarianceSum = (((covarianceCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      scratchA[sampleIndex] = (timeDeviation * timeDeviation);
    }
    unsigned int timeVarianceActive = sampleCount;
    unsigned int timeVarianceCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < timeVarianceActive) {
        unsigned int pairs = (timeVarianceActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((timeVarianceCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeVarianceCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeVarianceCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (timeVarianceActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (timeVarianceActive - 1u);
          double carryValue = (((timeVarianceCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeVarianceCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeVarianceActive = (pairs + odd);
        timeVarianceCurrentA = (((timeVarianceCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double timeVarianceSum = (((timeVarianceCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double driftSlope = (covarianceSum / timeVarianceSum);
    firstTimeOut[0u] = firstTime;
    lastTimeOut[0u] = lastTime;
    meanOut[0u] = mean;
    sampleStandardDeviationOut[0u] = sampleStandardDeviation;
    minimumOut[0u] = minimum;
    maximumOut[0u] = maximum;
    driftSlopeOut[0u] = driftSlope;
  }
}

)kernel";
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64(const double* volumeFlux, const double* faceArea, const double* maximumIncidentFaceSpeed, const double* interfaceNormalDotArea, const double* alphaFace, const double* compressionDatum, const unsigned int* nFaces, double* compressionSpeedOut, double* compressionFluxOut, double* alphaCompressionFluxOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int faceCount = nFaces[0];
  if (gid < faceCount) {
    double coefficient = compressionDatum[0];
    double alpha = alphaFace[gid];
    double compressionSpeed = ((((coefficient * maximumIncidentFaceSpeed[gid]) < ((coefficient * ((((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid]))) ? ((coefficient * maximumIncidentFaceSpeed[gid])) : (((coefficient * ((((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid])));
    double compressionFlux = (compressionSpeed * interfaceNormalDotArea[gid]);
    double alphaCompressionFlux = ((compressionFlux * alpha) * (__longlong_as_double(0x3ff0000000000000ull) - alpha));
    compressionSpeedOut[gid] = compressionSpeed;
    compressionFluxOut[gid] = compressionFlux;
    alphaCompressionFluxOut[gid] = alphaCompressionFlux;
  }
}

)kernel";
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64(const double* density, const double* volume, const double* streamwiseCoordinate, const double* acceptedVelocityX, const double* acceptedVelocityY, const double* acceptedVelocityZ, const double* dampingDatum, const unsigned int* nCells, double* lambdaOut, double* implicitDiagonalOut, double* implicitTargetRhsXOut, double* implicitTargetRhsYOut, double* implicitTargetRhsZOut, double* momentumSourceXOut, double* momentumSourceYOut, double* momentumSourceZOut, double* signedPowerOut, double* signedWorkOut, unsigned int* activeOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int cellCount = ((int)(nCells[0]));
  if (gid < cellCount) {
    double rawXi = ((streamwiseCoordinate[gid] - dampingDatum[1]) / (dampingDatum[2] - dampingDatum[1]));
    double xi = (((rawXi < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((__longlong_as_double(0x3ff0000000000000ull) < rawXi)) ? (__longlong_as_double(0x3ff0000000000000ull)) : (rawXi))));
    double ramp = ((xi * xi) * (__longlong_as_double(0x4008000000000000ull) - (__longlong_as_double(0x4000000000000000ull) * xi)));
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
    activeOut[gid] = (((__longlong_as_double(0x0000000000000000ull) < lambda)) ? (1u) : (0u));
  }
}

)kernel";
const char* k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64(const double* pointX, const double* pointY, const double* pointZ, const double* liquidFraction, const double* datum, const unsigned int* nCells, double* densityOut, double* pRghOut, double* absolutePressureOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int cellCount = ((int)(nCells[0]));
  if (gid < cellCount) {
    double alpha = liquidFraction[gid];
    double density = ((alpha * datum[10]) + ((__longlong_as_double(0x3ff0000000000000ull) - alpha) * datum[11]));
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
const char* k_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64(const double* alpha1, const double* k, const double* omega, const double* wallDistance, const double* gradientDotProduct, const double* strainMagnitudeSquared2, const double* datum, const unsigned int* nCells, double* densityOut, double* dynamicViscosityOut, double* kinematicViscosityOut, double* crossDiffusionOut, double* positiveCrossDiffusionOut, double* argument1Out, double* argument2Out, double* f1Out, double* f2Out, double* turbulentKinematicViscosityOut, double* turbulentDynamicViscosityOut, double* effectiveDynamicViscosityOut, double* nutToNuOut, double* gammaOut, double* betaOut, double* sigmaKOut, double* sigmaOmegaOut, double* productionRawOut, double* productionLimitedOut, double* kExplicitSourceOut, double* kImplicitSinkOut, double* omegaBaseSourceOut, double* crossLagOut, double* omegaCrossExplicitSourceOut, double* omegaCrossImplicitSinkOut, double* omegaImplicitSinkOut, double* kEffectiveDiffusivityOut, double* omegaEffectiveDiffusivityOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int cellCount = nCells[0];
  if (gid < cellCount) {
    double alpha = alpha1[gid];
    double kValue = k[gid];
    double omegaValue = omega[gid];
    double distance = wallDistance[gid];
    double gradientDot = gradientDotProduct[gid];
    double strain2 = strainMagnitudeSquared2[gid];
    double oneMinusAlpha = (__longlong_as_double(0x3ff0000000000000ull) - alpha);
    double density = ((alpha * datum[0]) + (oneMinusAlpha * datum[1]));
    double dynamicViscosity = ((alpha * datum[2]) + (oneMinusAlpha * datum[3]));
    double kinematicViscosity = (dynamicViscosity / density);
    double crossDiffusion = (((__longlong_as_double(0x4000000000000000ull) * __longlong_as_double(0x3feb645a1cac0831ull)) / omegaValue) * gradientDot);
    double positiveCrossDiffusion = (((crossDiffusion < __longlong_as_double(0x3ddb7cdfd9d7bdbbull))) ? (__longlong_as_double(0x3ddb7cdfd9d7bdbbull)) : (crossDiffusion));
    double rootK = sqrt(kValue);
    double distanceSquared = (distance * distance);
    double firstA = (rootK / ((__longlong_as_double(0x3fb70a3d70a3d70aull) * omegaValue) * distance));
    double firstB = ((__longlong_as_double(0x407f400000000000ull) * kinematicViscosity) / (distanceSquared * omegaValue));
    double firstC = (((__longlong_as_double(0x4010000000000000ull) * __longlong_as_double(0x3feb645a1cac0831ull)) * kValue) / (positiveCrossDiffusion * distanceSquared));
    double argument1 = (((firstC < (((firstA < firstB)) ? (firstB) : (firstA)))) ? (firstC) : ((((firstA < firstB)) ? (firstB) : (firstA))));
    double argument2 = ((((__longlong_as_double(0x4000000000000000ull) * firstA) < firstB)) ? (firstB) : ((__longlong_as_double(0x4000000000000000ull) * firstA)));
    double f1Value = (((((argument1 * argument1) * (argument1 * argument1)) < __longlong_as_double(0x4034000000000000ull))) ? (tanh(((argument1 * argument1) * (argument1 * argument1)))) : (__longlong_as_double(0x3ff0000000000000ull)));
    double f2Value = ((((argument2 * argument2) < __longlong_as_double(0x4034000000000000ull))) ? (tanh((argument2 * argument2))) : (__longlong_as_double(0x3ff0000000000000ull)));
    double strainMagnitude = sqrt(strain2);
    double nutDenominator = ((((__longlong_as_double(0x3fd3d70a3d70a3d7ull) * omegaValue) < ((__longlong_as_double(0x3ff0000000000000ull) * f2Value) * strainMagnitude))) ? (((__longlong_as_double(0x3ff0000000000000ull) * f2Value) * strainMagnitude)) : ((__longlong_as_double(0x3fd3d70a3d70a3d7ull) * omegaValue)));
    double nut = ((__longlong_as_double(0x3fd3d70a3d70a3d7ull) * kValue) / nutDenominator);
    double turbulentDynamicViscosity = (density * nut);
    double effectiveDynamicViscosity = (dynamicViscosity + turbulentDynamicViscosity);
    double nutToNu = (nut / (dynamicViscosity / density));
    double oneMinusF1 = (__longlong_as_double(0x3ff0000000000000ull) - f1Value);
    double gamma = ((f1Value * __longlong_as_double(0x3fe1c71c71c71c72ull)) + (oneMinusF1 * __longlong_as_double(0x3fdc28f5c28f5c29ull)));
    double beta = ((f1Value * __longlong_as_double(0x3fb3333333333333ull)) + (oneMinusF1 * __longlong_as_double(0x3fb532617c1bda51ull)));
    double sigmaK = ((f1Value * __longlong_as_double(0x3feb333333333333ull)) + (oneMinusF1 * __longlong_as_double(0x3ff0000000000000ull)));
    double sigmaOmega = ((f1Value * __longlong_as_double(0x3fe0000000000000ull)) + (oneMinusF1 * __longlong_as_double(0x3feb645a1cac0831ull)));
    double productionRaw = (turbulentDynamicViscosity * strain2);
    double productionLimit = ((((__longlong_as_double(0x4024000000000000ull) * __longlong_as_double(0x3fb70a3d70a3d70aull)) * density) * kValue) * omegaValue);
    double productionLimited = (((productionLimit < productionRaw)) ? (productionLimit) : (productionRaw));
    double kExplicitSource = (((productionLimited < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (productionLimited));
    double kImplicitSink = ((__longlong_as_double(0x3fb70a3d70a3d70aull) * density) * omegaValue);
    double omegaBaseSource = ((gamma * density) * strain2);
    double admittedOmega = (((omegaValue < __longlong_as_double(0x3d719799812dea11ull))) ? (__longlong_as_double(0x3d719799812dea11ull)) : (omegaValue));
    double crossLag = (((((__longlong_as_double(0x4000000000000000ull) * oneMinusF1) * density) * __longlong_as_double(0x3feb645a1cac0831ull)) / admittedOmega) * gradientDot);
    double omegaCrossExplicitSource = (((crossLag < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (crossLag));
    double omegaCrossImplicitSink = (((((-crossLag) < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((-crossLag))) / admittedOmega);
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
const char* k_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64 = R"kernel(
#include <cuda_runtime.h>
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

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64",
  9,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64[] = {
  { "alphaCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaMinimum", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaMaximum", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "incidentOffsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradientDotDisplacements", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "psiOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "limitingFaceOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64",
  8,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64",
  25,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64",
  25,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64",
  26,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64",
  18,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64",
  12,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64",
  10,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64",
  19,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64 = {
  1,
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64",
  9,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64 = {
  1,
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64",
  36,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64
};

const KernelArgumentInfo kAbiArgs_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64 = {
  1,
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "cuda",
  "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "kernel:cuda:Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  "abi-r1:cuda:Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64",
  28,
  kAbiArgs_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64
};

bool tryGetKernelAbiManifest_cuda_unmapped(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64;
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64") {
    out = &kAbiManifest_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_unmapped(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64";
    std::string_view sv(k_cuda_Solver_CFD_StandaloneCfd03SstGpuKernels_sstSpaldingWallFacesF64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

