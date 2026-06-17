// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_sparse_knn` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_sparse_knn_sparse_k_n_n_distance_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_sparse_k_n_n_distance_f32(const float* queryData, const int* queryIndices, const unsigned int* queryStart, const unsigned int* queryEnd, const float* idxData, const int* idxIndices, const int* idxIndptr, const unsigned int* nIdxRows, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nRows = nIdxRows[0];
  unsigned int qStart = queryStart[0];
  unsigned int qEnd = queryEnd[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    unsigned int rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    unsigned int idxStart = ((unsigned int)(idxRowStart));
    unsigned int idxEnd = ((unsigned int)(idxRowEnd));
    float queryNormSqAccum = __uint_as_float(0x00000000u);
    unsigned int qIdxAccum = qStart;
    unsigned int qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      unsigned int qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        float qVal = queryData[qPos];
        float qValSq = (qVal * qVal);
        float currQNorm = queryNormSqAccum;
        float newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      unsigned int currQIdx = qIdxAccum;
      unsigned int nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float queryNormSq = queryNormSqAccum;
    float idxNormSqAccum = __uint_as_float(0x00000000u);
    unsigned int iIdxAccum = idxStart;
    unsigned int idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      unsigned int iPos = iIdxAccum;
      if ((iPos < idxEnd)) {
        float iVal = idxData[iPos];
        float iValSq = (iVal * iVal);
        float currINorm = idxNormSqAccum;
        float newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      unsigned int currIIdx = iIdxAccum;
      unsigned int nextIIdx = (currIIdx + 1u);
      iIdxAccum = nextIIdx;
    }
    float idxNormSq = idxNormSqAccum;
    float dotAccum = __uint_as_float(0x00000000u);
    unsigned int qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      unsigned int qPos2 = qIdx2Accum;
      if ((qPos2 < qEnd)) {
        float qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        unsigned int iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          unsigned int iPos2 = iIdx2Accum;
          if ((iPos2 < idxEnd)) {
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
          unsigned int currI2Idx = iIdx2Accum;
          unsigned int nextI2Idx = (currI2Idx + 1u);
          iIdx2Accum = nextI2Idx;
        }
      }
      unsigned int currQ2Idx = qIdx2Accum;
      unsigned int nextQ2Idx = (currQ2Idx + 1u);
      qIdx2Accum = nextQ2Idx;
    }
    float dotProd = dotAccum;
    float twoDot = (__uint_as_float(0x40000000u) * dotProd);
    float sumNorms = (queryNormSq + idxNormSq);
    float distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}

)kernel";
const char* k_hip_navatala_sparse_knn_sparse_k_n_n_distance_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_sparse_k_n_n_distance_f64(const double* queryData, const int* queryIndices, const unsigned int* queryStart, const unsigned int* queryEnd, const double* idxData, const int* idxIndices, const int* idxIndptr, const unsigned int* nIdxRows, double* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nRows = nIdxRows[0];
  unsigned int qStart = queryStart[0];
  unsigned int qEnd = queryEnd[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    unsigned int rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    unsigned int idxStart = ((unsigned int)(idxRowStart));
    unsigned int idxEnd = ((unsigned int)(idxRowEnd));
    double queryNormSqAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int qIdxAccum = qStart;
    unsigned int qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      unsigned int qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        double qVal = queryData[qPos];
        double qValSq = (qVal * qVal);
        double currQNorm = queryNormSqAccum;
        double newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      unsigned int currQIdx = qIdxAccum;
      unsigned int nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    double queryNormSq = queryNormSqAccum;
    double idxNormSqAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int iIdxAccum = idxStart;
    unsigned int idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      unsigned int iPos = iIdxAccum;
      if ((iPos < idxEnd)) {
        double iVal = idxData[iPos];
        double iValSq = (iVal * iVal);
        double currINorm = idxNormSqAccum;
        double newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      unsigned int currIIdx = iIdxAccum;
      unsigned int nextIIdx = (currIIdx + 1u);
      iIdxAccum = nextIIdx;
    }
    double idxNormSq = idxNormSqAccum;
    double dotAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      unsigned int qPos2 = qIdx2Accum;
      if ((qPos2 < qEnd)) {
        double qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        unsigned int iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          unsigned int iPos2 = iIdx2Accum;
          if ((iPos2 < idxEnd)) {
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
          unsigned int currI2Idx = iIdx2Accum;
          unsigned int nextI2Idx = (currI2Idx + 1u);
          iIdx2Accum = nextI2Idx;
        }
      }
      unsigned int currQ2Idx = qIdx2Accum;
      unsigned int nextQ2Idx = (currQ2Idx + 1u);
      qIdx2Accum = nextQ2Idx;
    }
    double dotProd = dotAccum;
    double twoDot = (__longlong_as_double(0x4000000000000000ull) * dotProd);
    double sumNorms = (queryNormSq + idxNormSq);
    double distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}

)kernel";
const char* k_hip_navatala_sparse_knn_sparse_inner_product_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_sparse_inner_product_f32(const float* queryData, const int* queryIndices, const unsigned int* queryStart, const unsigned int* queryEnd, const float* idxData, const int* idxIndices, const int* idxIndptr, const unsigned int* nIdxRows, float* dotProducts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nRows = nIdxRows[0];
  unsigned int qStart = queryStart[0];
  unsigned int qEnd = queryEnd[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    unsigned int rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    unsigned int idxStart = ((unsigned int)(idxRowStart));
    unsigned int idxEnd = ((unsigned int)(idxRowEnd));
    float dotAccum = __uint_as_float(0x00000000u);
    unsigned int qIdxAccum = qStart;
    unsigned int qLen = (qEnd - qStart);
    unsigned int idxLen = (idxEnd - idxStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      unsigned int qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        float qVal = queryData[qPos];
        int qCol = queryIndices[qPos];
        unsigned int iIdxAccum = idxStart;
        for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
          unsigned int iPos = iIdxAccum;
          if ((iPos < idxEnd)) {
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
          unsigned int currIIdx = iIdxAccum;
          unsigned int nextIIdx = (currIIdx + 1u);
          iIdxAccum = nextIIdx;
        }
      }
      unsigned int currQIdx = qIdxAccum;
      unsigned int nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float dotProd = dotAccum;
    dotProducts[rowIdx] = dotProd;
  }
}

)kernel";
const char* k_hip_navatala_sparse_knn_sparse_inner_product_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_sparse_inner_product_f64(const double* queryData, const int* queryIndices, const unsigned int* queryStart, const unsigned int* queryEnd, const double* idxData, const int* idxIndices, const int* idxIndptr, const unsigned int* nIdxRows, double* dotProducts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nRows = nIdxRows[0];
  unsigned int qStart = queryStart[0];
  unsigned int qEnd = queryEnd[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    unsigned int rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    unsigned int idxStart = ((unsigned int)(idxRowStart));
    unsigned int idxEnd = ((unsigned int)(idxRowEnd));
    double dotAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int qIdxAccum = qStart;
    unsigned int qLen = (qEnd - qStart);
    unsigned int idxLen = (idxEnd - idxStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      unsigned int qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        double qVal = queryData[qPos];
        int qCol = queryIndices[qPos];
        unsigned int iIdxAccum = idxStart;
        for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
          unsigned int iPos = iIdxAccum;
          if ((iPos < idxEnd)) {
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
          unsigned int currIIdx = iIdxAccum;
          unsigned int nextIIdx = (currIIdx + 1u);
          iIdxAccum = nextIIdx;
        }
      }
      unsigned int currQIdx = qIdxAccum;
      unsigned int nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    double dotProd = dotAccum;
    dotProducts[rowIdx] = dotProd;
  }
}

)kernel";
const char* k_hip_navatala_sparse_knn_sparse_top_k_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_sparse_top_k_f32(const float* distances, const unsigned int* nIdxRows, const unsigned int* k, float* outDistances, unsigned int* outIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    unsigned int nRows = nIdxRows[0];
    unsigned int kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      unsigned int initIdxU32 = ((unsigned int)(initIdx));
      outDistances[initIdxU32] = __uint_as_float(0x7f7fc99eu);
      outIndices[initIdxU32] = 4294967295u;
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      float dist = distances[iU32];
      unsigned int lastIdx = (kVal - 1u);
      float kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        unsigned int insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          unsigned int scanIdxU32 = ((unsigned int)(scanIdx));
          unsigned int currentInsertPos = insertPosAccum;
          unsigned int checkPos = (lastIdx - scanIdxU32);
          float checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          unsigned int newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        unsigned int finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          unsigned int shiftIdxU32 = ((unsigned int)(shiftIdx));
          unsigned int shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            unsigned int srcPos = (shiftPos - 1u);
            float srcDist = outDistances[srcPos];
            unsigned int srcIdx = outIndices[srcPos];
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
const char* k_hip_navatala_sparse_knn_sparse_top_k_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_sparse_top_k_f64(const double* distances, const unsigned int* nIdxRows, const unsigned int* k, double* outDistances, unsigned int* outIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    unsigned int nRows = nIdxRows[0];
    unsigned int kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      unsigned int initIdxU32 = ((unsigned int)(initIdx));
      outDistances[initIdxU32] = __longlong_as_double(0x7fee42d130773b76ull);
      outIndices[initIdxU32] = 4294967295u;
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      double dist = distances[iU32];
      unsigned int lastIdx = (kVal - 1u);
      double kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        unsigned int insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          unsigned int scanIdxU32 = ((unsigned int)(scanIdx));
          unsigned int currentInsertPos = insertPosAccum;
          unsigned int checkPos = (lastIdx - scanIdxU32);
          double checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          unsigned int newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        unsigned int finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          unsigned int shiftIdxU32 = ((unsigned int)(shiftIdx));
          unsigned int shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            unsigned int srcPos = (shiftPos - 1u);
            double srcDist = outDistances[srcPos];
            unsigned int srcIdx = outIndices[srcPos];
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
const char* k_hip_navatala_sparse_knn_csr_row_distance_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_csr_row_distance_f32(const float* sparseData, const int* sparseIndices, const unsigned int* sparseStart, const unsigned int* sparseEnd, const float* denseQuery, const unsigned int* nCols, float* distance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int sStart = sparseStart[0];
  unsigned int sEnd = sparseEnd[0];
  unsigned int nColsVal = nCols[0];
  __shared__ float sdata[256];
  float sparseNormAccum = __uint_as_float(0x00000000u);
  float dotProdAccum = __uint_as_float(0x00000000u);
  unsigned int sIdxAccum = sStart;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  unsigned int sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    unsigned int sPos = sIdxAccum;
    if ((sPos < sEnd)) {
      float sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      unsigned int sColU32 = ((unsigned int)(sCol));
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
    unsigned int currSIdx = sIdxAccum;
    unsigned int nextSIdx = (currSIdx + 1u);
    sIdxAccum = nextSIdx;
  }
  float denseNormAccum = __uint_as_float(0x00000000u);
  unsigned int dIdxAccum = 0u;
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    unsigned int dPos = dIdxAccum;
    if ((dPos < nColsVal)) {
      float dValLoop = denseQuery[dPos];
      float dValSq = (dValLoop * dValLoop);
      float currDNorm = denseNormAccum;
      float newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    unsigned int currDIdx = dIdxAccum;
    unsigned int nextDIdx = (currDIdx + 1u);
    dIdxAccum = nextDIdx;
  }
  float sparseNormSq = sparseNormAccum;
  float denseNormSq = denseNormAccum;
  float dotProd = dotProdAccum;
  float twoDot = (__uint_as_float(0x40000000u) * dotProd);
  float sumNorms = (sparseNormSq + denseNormSq);
  float distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    distance[0u] = distSq;
  }
}

)kernel";
const char* k_hip_navatala_sparse_knn_csr_row_distance_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_knn_csr_row_distance_f64(const double* sparseData, const int* sparseIndices, const unsigned int* sparseStart, const unsigned int* sparseEnd, const double* denseQuery, const unsigned int* nCols, double* distance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int sStart = sparseStart[0];
  unsigned int sEnd = sparseEnd[0];
  unsigned int nColsVal = nCols[0];
  __shared__ double sdata[256];
  double sparseNormAccum = __longlong_as_double(0x0000000000000000ull);
  double dotProdAccum = __longlong_as_double(0x0000000000000000ull);
  unsigned int sIdxAccum = sStart;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  unsigned int sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    unsigned int sPos = sIdxAccum;
    if ((sPos < sEnd)) {
      double sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      unsigned int sColU32 = ((unsigned int)(sCol));
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
    unsigned int currSIdx = sIdxAccum;
    unsigned int nextSIdx = (currSIdx + 1u);
    sIdxAccum = nextSIdx;
  }
  double denseNormAccum = __longlong_as_double(0x0000000000000000ull);
  unsigned int dIdxAccum = 0u;
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    unsigned int dPos = dIdxAccum;
    if ((dPos < nColsVal)) {
      double dValLoop = denseQuery[dPos];
      double dValSq = (dValLoop * dValLoop);
      double currDNorm = denseNormAccum;
      double newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    unsigned int currDIdx = dIdxAccum;
    unsigned int nextDIdx = (currDIdx + 1u);
    dIdxAccum = nextDIdx;
  }
  double sparseNormSq = sparseNormAccum;
  double denseNormSq = denseNormAccum;
  double dotProd = dotProdAccum;
  double twoDot = (__longlong_as_double(0x4000000000000000ull) * dotProd);
  double sumNorms = (sparseNormSq + denseNormSq);
  double distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    distance[0u] = distSq;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_sparse_k_n_n_distance_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_sparse_k_n_n_distance_f32 = {
  1,
  "navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "hip",
  "navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "kernel:hip:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "abi-r1:hip:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "abi-r1:hip:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  9,
  kAbiArgs_hip_navatala_sparse_knn_sparse_k_n_n_distance_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_sparse_k_n_n_distance_f64[] = {
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
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_sparse_k_n_n_distance_f64 = {
  1,
  "navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "hip",
  "navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "kernel:hip:navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "abi-r1:hip:navatala_sparse_knn_sparse_k_n_n_distance_f64",
  "abi-r1:hip:navatala_sparse_knn_sparse_k_n_n_distance_f64",
  9,
  kAbiArgs_hip_navatala_sparse_knn_sparse_k_n_n_distance_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_sparse_inner_product_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_sparse_inner_product_f32 = {
  1,
  "navatala_sparse_knn_sparse_inner_product_f32",
  "hip",
  "navatala_sparse_knn_sparse_inner_product_f32",
  "kernel:hip:navatala_sparse_knn_sparse_inner_product_f32",
  "abi-r1:hip:navatala_sparse_knn_sparse_inner_product_f32",
  "abi-r1:hip:navatala_sparse_knn_sparse_inner_product_f32",
  9,
  kAbiArgs_hip_navatala_sparse_knn_sparse_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_sparse_inner_product_f64[] = {
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
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_sparse_inner_product_f64 = {
  1,
  "navatala_sparse_knn_sparse_inner_product_f64",
  "hip",
  "navatala_sparse_knn_sparse_inner_product_f64",
  "kernel:hip:navatala_sparse_knn_sparse_inner_product_f64",
  "abi-r1:hip:navatala_sparse_knn_sparse_inner_product_f64",
  "abi-r1:hip:navatala_sparse_knn_sparse_inner_product_f64",
  9,
  kAbiArgs_hip_navatala_sparse_knn_sparse_inner_product_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_sparse_top_k_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDistances", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outIndices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_sparse_top_k_f32 = {
  1,
  "navatala_sparse_knn_sparse_top_k_f32",
  "hip",
  "navatala_sparse_knn_sparse_top_k_f32",
  "kernel:hip:navatala_sparse_knn_sparse_top_k_f32",
  "abi-r1:hip:navatala_sparse_knn_sparse_top_k_f32",
  "abi-r1:hip:navatala_sparse_knn_sparse_top_k_f32",
  5,
  kAbiArgs_hip_navatala_sparse_knn_sparse_top_k_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_sparse_top_k_f64[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDistances", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outIndices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_sparse_top_k_f64 = {
  1,
  "navatala_sparse_knn_sparse_top_k_f64",
  "hip",
  "navatala_sparse_knn_sparse_top_k_f64",
  "kernel:hip:navatala_sparse_knn_sparse_top_k_f64",
  "abi-r1:hip:navatala_sparse_knn_sparse_top_k_f64",
  "abi-r1:hip:navatala_sparse_knn_sparse_top_k_f64",
  5,
  kAbiArgs_hip_navatala_sparse_knn_sparse_top_k_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_csr_row_distance_f32[] = {
  { "sparseData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sparseEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "denseQuery", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distance", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_csr_row_distance_f32 = {
  1,
  "navatala_sparse_knn_csr_row_distance_f32",
  "hip",
  "navatala_sparse_knn_csr_row_distance_f32",
  "kernel:hip:navatala_sparse_knn_csr_row_distance_f32",
  "abi-r1:hip:navatala_sparse_knn_csr_row_distance_f32",
  "abi-r1:hip:navatala_sparse_knn_csr_row_distance_f32",
  7,
  kAbiArgs_hip_navatala_sparse_knn_csr_row_distance_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_sparse_knn_csr_row_distance_f64[] = {
  { "sparseData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sparseEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "denseQuery", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distance", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_sparse_knn_csr_row_distance_f64 = {
  1,
  "navatala_sparse_knn_csr_row_distance_f64",
  "hip",
  "navatala_sparse_knn_csr_row_distance_f64",
  "kernel:hip:navatala_sparse_knn_csr_row_distance_f64",
  "abi-r1:hip:navatala_sparse_knn_csr_row_distance_f64",
  "abi-r1:hip:navatala_sparse_knn_csr_row_distance_f64",
  7,
  kAbiArgs_hip_navatala_sparse_knn_csr_row_distance_f64
};

bool tryGetKernelAbiManifest_hip_sparse_knn(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f32") {
    out = &kAbiManifest_hip_navatala_sparse_knn_sparse_k_n_n_distance_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f64") {
    out = &kAbiManifest_hip_navatala_sparse_knn_sparse_k_n_n_distance_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_inner_product_f32") {
    out = &kAbiManifest_hip_navatala_sparse_knn_sparse_inner_product_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_inner_product_f64") {
    out = &kAbiManifest_hip_navatala_sparse_knn_sparse_inner_product_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_top_k_f32") {
    out = &kAbiManifest_hip_navatala_sparse_knn_sparse_top_k_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_top_k_f64") {
    out = &kAbiManifest_hip_navatala_sparse_knn_sparse_top_k_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_csr_row_distance_f32") {
    out = &kAbiManifest_hip_navatala_sparse_knn_csr_row_distance_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_csr_row_distance_f64") {
    out = &kAbiManifest_hip_navatala_sparse_knn_csr_row_distance_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_sparse_knn(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_sparse_k_n_n_distance_f32";
    std::string_view sv(k_hip_navatala_sparse_knn_sparse_k_n_n_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_sparse_k_n_n_distance_f64";
    std::string_view sv(k_hip_navatala_sparse_knn_sparse_k_n_n_distance_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_sparse_inner_product_f32";
    std::string_view sv(k_hip_navatala_sparse_knn_sparse_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_inner_product_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_sparse_inner_product_f64";
    std::string_view sv(k_hip_navatala_sparse_knn_sparse_inner_product_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_sparse_top_k_f32";
    std::string_view sv(k_hip_navatala_sparse_knn_sparse_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_sparse_top_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_sparse_top_k_f64";
    std::string_view sv(k_hip_navatala_sparse_knn_sparse_top_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_csr_row_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_csr_row_distance_f32";
    std::string_view sv(k_hip_navatala_sparse_knn_csr_row_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_sparse_knn_csr_row_distance_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_sparse_knn_csr_row_distance_f64";
    std::string_view sv(k_hip_navatala_sparse_knn_csr_row_distance_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

