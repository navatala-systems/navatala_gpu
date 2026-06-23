// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_sparse_knn` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f32 = R"kernel(
__kernel void navatala_sparse_knn_sparse_k_n_n_distance_f32(__global const float* queryData, __global const int* queryIndices, __global const uint* queryStart, __global const uint* queryEnd, __global const float* idxData, __global const int* idxIndices, __global const int* idxIndptr, __global const uint* nIdxRows, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + (uint)(1u));
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    float queryNormSqAccum = as_float(0x00000000u);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if (qPos < qEnd) {
        float qVal = queryData[qPos];
        float qValSq = (qVal * qVal);
        float currQNorm = queryNormSqAccum;
        float newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + (uint)(1u));
      qIdxAccum = nextQIdx;
    }
    float queryNormSq = queryNormSqAccum;
    float idxNormSqAccum = as_float(0x00000000u);
    uint iIdxAccum = idxStart;
    uint idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      uint iPos = iIdxAccum;
      if (iPos < idxEnd) {
        float iVal = idxData[iPos];
        float iValSq = (iVal * iVal);
        float currINorm = idxNormSqAccum;
        float newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      uint currIIdx = iIdxAccum;
      uint nextIIdx = (currIIdx + (uint)(1u));
      iIdxAccum = nextIIdx;
    }
    float idxNormSq = idxNormSqAccum;
    float dotAccum = as_float(0x00000000u);
    uint qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      uint qPos2 = qIdx2Accum;
      if (qPos2 < qEnd) {
        float qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        uint iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          uint iPos2 = iIdx2Accum;
          if (iPos2 < idxEnd) {
            int iCol = idxIndices[iPos2];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              float iVal2 = idxData[iPos2];
              float prod = (qVal2 * iVal2);
              float currDot = dotAccum;
              float newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currI2Idx = iIdx2Accum;
          uint nextI2Idx = (currI2Idx + (uint)(1u));
          iIdx2Accum = nextI2Idx;
        }
      }
      uint currQ2Idx = qIdx2Accum;
      uint nextQ2Idx = (currQ2Idx + (uint)(1u));
      qIdx2Accum = nextQ2Idx;
    }
    float dotProd = dotAccum;
    float twoDot = (as_float(0x40000000u) * dotProd);
    float sumNorms = (queryNormSq + idxNormSq);
    float distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_knn_sparse_k_n_n_distance_f64(__global const double* queryData, __global const int* queryIndices, __global const uint* queryStart, __global const uint* queryEnd, __global const double* idxData, __global const int* idxIndices, __global const int* idxIndptr, __global const uint* nIdxRows, __global double* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + (uint)(1u));
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    double queryNormSqAccum = as_double(0x0000000000000000ul);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if (qPos < qEnd) {
        double qVal = queryData[qPos];
        double qValSq = (qVal * qVal);
        double currQNorm = queryNormSqAccum;
        double newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + (uint)(1u));
      qIdxAccum = nextQIdx;
    }
    double queryNormSq = queryNormSqAccum;
    double idxNormSqAccum = as_double(0x0000000000000000ul);
    uint iIdxAccum = idxStart;
    uint idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      uint iPos = iIdxAccum;
      if (iPos < idxEnd) {
        double iVal = idxData[iPos];
        double iValSq = (iVal * iVal);
        double currINorm = idxNormSqAccum;
        double newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      uint currIIdx = iIdxAccum;
      uint nextIIdx = (currIIdx + (uint)(1u));
      iIdxAccum = nextIIdx;
    }
    double idxNormSq = idxNormSqAccum;
    double dotAccum = as_double(0x0000000000000000ul);
    uint qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      uint qPos2 = qIdx2Accum;
      if (qPos2 < qEnd) {
        double qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        uint iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          uint iPos2 = iIdx2Accum;
          if (iPos2 < idxEnd) {
            int iCol = idxIndices[iPos2];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              double iVal2 = idxData[iPos2];
              double prod = (qVal2 * iVal2);
              double currDot = dotAccum;
              double newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currI2Idx = iIdx2Accum;
          uint nextI2Idx = (currI2Idx + (uint)(1u));
          iIdx2Accum = nextI2Idx;
        }
      }
      uint currQ2Idx = qIdx2Accum;
      uint nextQ2Idx = (currQ2Idx + (uint)(1u));
      qIdx2Accum = nextQ2Idx;
    }
    double dotProd = dotAccum;
    double twoDot = (as_double(0x4000000000000000ul) * dotProd);
    double sumNorms = (queryNormSq + idxNormSq);
    double distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_sparse_inner_product_f32 = R"kernel(
__kernel void navatala_sparse_knn_sparse_inner_product_f32(__global const float* queryData, __global const int* queryIndices, __global const uint* queryStart, __global const uint* queryEnd, __global const float* idxData, __global const int* idxIndices, __global const int* idxIndptr, __global const uint* nIdxRows, __global float* dotProducts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + (uint)(1u));
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    float dotAccum = as_float(0x00000000u);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    uint idxLen = (idxEnd - idxStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if (qPos < qEnd) {
        float qVal = queryData[qPos];
        int qCol = queryIndices[qPos];
        uint iIdxAccum = idxStart;
        for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
          uint iPos = iIdxAccum;
          if (iPos < idxEnd) {
            int iCol = idxIndices[iPos];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              float iVal = idxData[iPos];
              float prod = (qVal * iVal);
              float currDot = dotAccum;
              float newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currIIdx = iIdxAccum;
          uint nextIIdx = (currIIdx + (uint)(1u));
          iIdxAccum = nextIIdx;
        }
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + (uint)(1u));
      qIdxAccum = nextQIdx;
    }
    float dotProd = dotAccum;
    dotProducts[rowIdx] = dotProd;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_sparse_inner_product_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_knn_sparse_inner_product_f64(__global const double* queryData, __global const int* queryIndices, __global const uint* queryStart, __global const uint* queryEnd, __global const double* idxData, __global const int* idxIndices, __global const int* idxIndptr, __global const uint* nIdxRows, __global double* dotProducts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + (uint)(1u));
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    double dotAccum = as_double(0x0000000000000000ul);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    uint idxLen = (idxEnd - idxStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if (qPos < qEnd) {
        double qVal = queryData[qPos];
        int qCol = queryIndices[qPos];
        uint iIdxAccum = idxStart;
        for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
          uint iPos = iIdxAccum;
          if (iPos < idxEnd) {
            int iCol = idxIndices[iPos];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              double iVal = idxData[iPos];
              double prod = (qVal * iVal);
              double currDot = dotAccum;
              double newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currIIdx = iIdxAccum;
          uint nextIIdx = (currIIdx + (uint)(1u));
          iIdxAccum = nextIIdx;
        }
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + (uint)(1u));
      qIdxAccum = nextQIdx;
    }
    double dotProd = dotAccum;
    dotProducts[rowIdx] = dotProd;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_sparse_top_k_f32 = R"kernel(
__kernel void navatala_sparse_knn_sparse_top_k_f32(__global const float* distances, __global const uint* nIdxRows, __global const uint* k, __global float* outDistances, __global uint* outIndices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    uint nRows = nIdxRows[0];
    uint kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      outDistances[initIdxU32] = as_float(0x7f7fc99eu);
      outIndices[initIdxU32] = (uint)(4294967295u);
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      uint iU32 = ((uint)(i));
      float dist = distances[iU32];
      uint lastIdx = (kVal - (uint)(1u));
      float kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          float checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          uint newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            uint srcPos = (shiftPos - (uint)(1u));
            float srcDist = outDistances[srcPos];
            uint srcIdx = outIndices[srcPos];
            outDistances[shiftPos] = srcDist;
            outIndices[shiftPos] = srcIdx;
          }
        }
        outDistances[finalInsertPos] = dist;
        outIndices[finalInsertPos] = iU32;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_sparse_top_k_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_knn_sparse_top_k_f64(__global const double* distances, __global const uint* nIdxRows, __global const uint* k, __global double* outDistances, __global uint* outIndices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    uint nRows = nIdxRows[0];
    uint kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      outDistances[initIdxU32] = as_double(0x7fee42d130773b76ul);
      outIndices[initIdxU32] = (uint)(4294967295u);
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      uint iU32 = ((uint)(i));
      double dist = distances[iU32];
      uint lastIdx = (kVal - (uint)(1u));
      double kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          double checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          uint newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            uint srcPos = (shiftPos - (uint)(1u));
            double srcDist = outDistances[srcPos];
            uint srcIdx = outIndices[srcPos];
            outDistances[shiftPos] = srcDist;
            outIndices[shiftPos] = srcIdx;
          }
        }
        outDistances[finalInsertPos] = dist;
        outIndices[finalInsertPos] = iU32;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_csr_row_distance_f32 = R"kernel(
__kernel void navatala_sparse_knn_csr_row_distance_f32(__global const float* sparseData, __global const int* sparseIndices, __global const uint* sparseStart, __global const uint* sparseEnd, __global const float* denseQuery, __global const uint* nCols, __global float* distance) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint sStart = sparseStart[0];
  uint sEnd = sparseEnd[0];
  uint nColsVal = nCols[0];
  __local float sdata[256];
  float sparseNormAccum = as_float(0x00000000u);
  float dotProdAccum = as_float(0x00000000u);
  uint sIdxAccum = sStart;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  uint sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    uint sPos = sIdxAccum;
    if (sPos < sEnd) {
      float sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      uint sColU32 = ((uint)(sCol));
      float sValSq = (sVal * sVal);
      float currSNorm = sparseNormAccum;
      float newSNorm = (currSNorm + sValSq);
      sparseNormAccum = newSNorm;
      float dVal = denseQuery[sColU32];
      float prod = (sVal * dVal);
      float currDot = dotProdAccum;
      float newDot = (currDot + prod);
      dotProdAccum = newDot;
    }
    uint currSIdx = sIdxAccum;
    uint nextSIdx = (currSIdx + (uint)(1u));
    sIdxAccum = nextSIdx;
  }
  float denseNormAccum = as_float(0x00000000u);
  uint dIdxAccum = (uint)(0u);
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    uint dPos = dIdxAccum;
    if (dPos < nColsVal) {
      float dValLoop = denseQuery[dPos];
      float dValSq = (dValLoop * dValLoop);
      float currDNorm = denseNormAccum;
      float newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    uint currDIdx = dIdxAccum;
    uint nextDIdx = (currDIdx + (uint)(1u));
    dIdxAccum = nextDIdx;
  }
  float sparseNormSq = sparseNormAccum;
  float denseNormSq = denseNormAccum;
  float dotProd = dotProdAccum;
  float twoDot = (as_float(0x40000000u) * dotProd);
  float sumNorms = (sparseNormSq + denseNormSq);
  float distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    distance[(uint)(0u)] = distSq;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_knn_csr_row_distance_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_knn_csr_row_distance_f64(__global const double* sparseData, __global const int* sparseIndices, __global const uint* sparseStart, __global const uint* sparseEnd, __global const double* denseQuery, __global const uint* nCols, __global double* distance) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint sStart = sparseStart[0];
  uint sEnd = sparseEnd[0];
  uint nColsVal = nCols[0];
  __local double sdata[256];
  double sparseNormAccum = as_double(0x0000000000000000ul);
  double dotProdAccum = as_double(0x0000000000000000ul);
  uint sIdxAccum = sStart;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  uint sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    uint sPos = sIdxAccum;
    if (sPos < sEnd) {
      double sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      uint sColU32 = ((uint)(sCol));
      double sValSq = (sVal * sVal);
      double currSNorm = sparseNormAccum;
      double newSNorm = (currSNorm + sValSq);
      sparseNormAccum = newSNorm;
      double dVal = denseQuery[sColU32];
      double prod = (sVal * dVal);
      double currDot = dotProdAccum;
      double newDot = (currDot + prod);
      dotProdAccum = newDot;
    }
    uint currSIdx = sIdxAccum;
    uint nextSIdx = (currSIdx + (uint)(1u));
    sIdxAccum = nextSIdx;
  }
  double denseNormAccum = as_double(0x0000000000000000ul);
  uint dIdxAccum = (uint)(0u);
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    uint dPos = dIdxAccum;
    if (dPos < nColsVal) {
      double dValLoop = denseQuery[dPos];
      double dValSq = (dValLoop * dValLoop);
      double currDNorm = denseNormAccum;
      double newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    uint currDIdx = dIdxAccum;
    uint nextDIdx = (currDIdx + (uint)(1u));
    dIdxAccum = nextDIdx;
  }
  double sparseNormSq = sparseNormAccum;
  double denseNormSq = denseNormAccum;
  double dotProd = dotProdAccum;
  double twoDot = (as_double(0x4000000000000000ul) * dotProd);
  double sumNorms = (sparseNormSq + denseNormSq);
  double distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    distance[(uint)(0u)] = distSq;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f32[] = {
  { "queryData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "queryEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "idxData", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndptr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f32 = {
  1,
  "navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "opencl",
  "navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "kernel:opencl:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "abi-r1:opencl:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "abi-r1:opencl:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  9,
  kAbiArgs_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f64[] = {
  { "queryData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "queryEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "idxData", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndptr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f64 = {
  1,
  "navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "opencl",
  "navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "kernel:opencl:navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "abi-r1:opencl:navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "abi-r1:opencl:navatala_sparse_knn_sparse_k_n_n_distance_f64",
  9,
  kAbiArgs_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_sparse_inner_product_f32[] = {
  { "queryData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "queryEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "idxData", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndptr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotProducts", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_sparse_inner_product_f32 = {
  1,
  "navatala_sparse_knn_sparse_inner_product_f32",
  "opencl",
  "navatala_sparse_knn_sparse_inner_product_f32",
  "kernel:opencl:navatala_sparse_knn_sparse_inner_product_f32",
  "abi-r1:opencl:navatala_sparse_knn_sparse_inner_product_f32",
  "abi-r1:opencl:navatala_sparse_knn_sparse_inner_product_f32",
  9,
  kAbiArgs_opencl_navatala_sparse_knn_sparse_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_sparse_inner_product_f64[] = {
  { "queryData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "queryEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "idxData", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndptr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotProducts", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_sparse_inner_product_f64 = {
  1,
  "navatala_sparse_knn_sparse_inner_product_f64",
  "opencl",
  "navatala_sparse_knn_sparse_inner_product_f64",
  "kernel:opencl:navatala_sparse_knn_sparse_inner_product_f64",
  "abi-r1:opencl:navatala_sparse_knn_sparse_inner_product_f64",
  "abi-r1:opencl:navatala_sparse_knn_sparse_inner_product_f64",
  9,
  kAbiArgs_opencl_navatala_sparse_knn_sparse_inner_product_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_sparse_top_k_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDistances", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outIndices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_sparse_top_k_f32 = {
  1,
  "navatala_sparse_knn_sparse_top_k_f32",
  "opencl",
  "navatala_sparse_knn_sparse_top_k_f32",
  "kernel:opencl:navatala_sparse_knn_sparse_top_k_f32",
  "abi-r1:opencl:navatala_sparse_knn_sparse_top_k_f32",
  "abi-r1:opencl:navatala_sparse_knn_sparse_top_k_f32",
  5,
  kAbiArgs_opencl_navatala_sparse_knn_sparse_top_k_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_sparse_top_k_f64[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDistances", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outIndices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_sparse_top_k_f64 = {
  1,
  "navatala_sparse_knn_sparse_top_k_f64",
  "opencl",
  "navatala_sparse_knn_sparse_top_k_f64",
  "kernel:opencl:navatala_sparse_knn_sparse_top_k_f64",
  "abi-r1:opencl:navatala_sparse_knn_sparse_top_k_f64",
  "abi-r1:opencl:navatala_sparse_knn_sparse_top_k_f64",
  5,
  kAbiArgs_opencl_navatala_sparse_knn_sparse_top_k_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_csr_row_distance_f32[] = {
  { "sparseData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sparseEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "denseQuery", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distance", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_csr_row_distance_f32 = {
  1,
  "navatala_sparse_knn_csr_row_distance_f32",
  "opencl",
  "navatala_sparse_knn_csr_row_distance_f32",
  "kernel:opencl:navatala_sparse_knn_csr_row_distance_f32",
  "abi-r1:opencl:navatala_sparse_knn_csr_row_distance_f32",
  "abi-r1:opencl:navatala_sparse_knn_csr_row_distance_f32",
  7,
  kAbiArgs_opencl_navatala_sparse_knn_csr_row_distance_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_knn_csr_row_distance_f64[] = {
  { "sparseData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sparseEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "denseQuery", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distance", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_knn_csr_row_distance_f64 = {
  1,
  "navatala_sparse_knn_csr_row_distance_f64",
  "opencl",
  "navatala_sparse_knn_csr_row_distance_f64",
  "kernel:opencl:navatala_sparse_knn_csr_row_distance_f64",
  "abi-r1:opencl:navatala_sparse_knn_csr_row_distance_f64",
  "abi-r1:opencl:navatala_sparse_knn_csr_row_distance_f64",
  7,
  kAbiArgs_opencl_navatala_sparse_knn_csr_row_distance_f64
};

bool tryGetKernelAbiManifest_opencl_sparse_knn(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_inner_product_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_sparse_inner_product_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_inner_product_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_sparse_inner_product_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_top_k_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_sparse_top_k_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_top_k_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_sparse_top_k_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_csr_row_distance_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_csr_row_distance_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_csr_row_distance_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_knn_csr_row_distance_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_sparse_knn(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_sparse_k_n_n_distance_f32";
    std::string_view sv(k_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_sparse_k_n_n_distance_f64";
    std::string_view sv(k_opencl_navatala_sparse_knn_sparse_k_n_n_distance_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_sparse_inner_product_f32";
    std::string_view sv(k_opencl_navatala_sparse_knn_sparse_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_inner_product_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_sparse_inner_product_f64";
    std::string_view sv(k_opencl_navatala_sparse_knn_sparse_inner_product_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_sparse_top_k_f32";
    std::string_view sv(k_opencl_navatala_sparse_knn_sparse_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_sparse_top_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_sparse_top_k_f64";
    std::string_view sv(k_opencl_navatala_sparse_knn_sparse_top_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_csr_row_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_csr_row_distance_f32";
    std::string_view sv(k_opencl_navatala_sparse_knn_csr_row_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_knn_csr_row_distance_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_knn_csr_row_distance_f64";
    std::string_view sv(k_opencl_navatala_sparse_knn_csr_row_distance_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

