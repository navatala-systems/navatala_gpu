// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_sparse` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_sparse_sparse_add_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_add_f32(device const float* valuesA [[buffer(0)]], device const float* valuesB [[buffer(1)]], device const int* nnz [[buffer(2)]], device float* valuesC [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < nnz[0u]) {
    float a = valuesA[i];
    float b = valuesB[i];
    valuesC[i] = (a + b);
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_sub_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_sub_f32(device const float* valuesA [[buffer(0)]], device const float* valuesB [[buffer(1)]], device const int* nnz [[buffer(2)]], device float* valuesC [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < nnz[0u]) {
    float a = valuesA[i];
    float b = valuesB[i];
    valuesC[i] = (a - b);
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_scale_f32(device const float* valuesA [[buffer(0)]], device const float* alpha [[buffer(1)]], device const int* nnz [[buffer(2)]], device float* valuesB [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  float alphaVal = alpha[0u];
  if (i < nnz[0u]) {
    float a = valuesA[i];
    valuesB[i] = (alphaVal * a);
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_hadamard_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_hadamard_f32(device const float* valuesA [[buffer(0)]], device const float* valuesB [[buffer(1)]], device const int* nnz [[buffer(2)]], device float* valuesC [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < nnz[0u]) {
    float a = valuesA[i];
    float b = valuesB[i];
    valuesC[i] = (a * b);
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_abs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_abs_f32(device const float* valuesA [[buffer(0)]], device const int* nnz [[buffer(1)]], device float* valuesB [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < nnz[0u]) {
    float a = valuesA[i];
    valuesB[i] = abs(a);
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_threshold_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_threshold_f32(device const float* valuesA [[buffer(0)]], device const float* threshold [[buffer(1)]], device const int* nnz [[buffer(2)]], device float* valuesB [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  float thresh = threshold[0u];
  if (i < nnz[0u]) {
    float a = valuesA[i];
    float absA = abs(a);
    float result = (((absA >= thresh)) ? (a) : (as_type<float>(0x00000000u)));
    valuesB[i] = result;
  }
}

)kernel";
const char* k_metal_navatala_sparse_coo_transpose_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_coo_transpose_f32(device const uint* rowInd [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nnz [[buffer(3)]], device uint* rowInd_out [[buffer(4)]], device uint* colInd_out [[buffer(5)]], device float* values_out [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  if (i < nnz[0u]) {
    uint col = colInd[i];
    rowInd_out[i] = col;
    uint row = rowInd[i];
    colInd_out[i] = row;
    float val = values[i];
    values_out[i] = val;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_inner_product_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_inner_product_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* nnz [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lidU32 = ((uint)(int(__tid.x)));
  uint nnzVal = nnz[0u];
  float localSumAccum = as_type<float>(0x00000000u);
  uint idxAccum = gid;
  uint gridSize = ((uint)(int(__grid_size.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      float prod = (aVal * bVal);
      float currentSum = localSumAccum;
      float newSum = (currentSum + prod);
      localSumAccum = newSum;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  float finalLocalSum = localSumAccum;
  sdata[lidU32] = finalLocalSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if (lidU32 < currentStride) {
      uint otherIdx = (lidU32 + currentStride);
      float myVal = sdata[lidU32];
      float otherVal = sdata[otherIdx];
      sdata[lidU32] = (myVal + otherVal);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lidU32 == 0u) {
    result[0u] = sdata[0u];
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_l2_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_l2_distance_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* nnz [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lidU32 = ((uint)(int(__tid.x)));
  uint nnzVal = nnz[0u];
  float localSumSqAccum = as_type<float>(0x00000000u);
  uint idxAccum = gid;
  uint gridSize = ((uint)(int(__grid_size.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      float diff = (aVal - bVal);
      float diffSq = (diff * diff);
      float currentSumSq = localSumSqAccum;
      float newSumSq = (currentSumSq + diffSq);
      localSumSqAccum = newSumSq;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  float finalLocalSumSq = localSumSqAccum;
  sdata[lidU32] = finalLocalSumSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if (lidU32 < currentStride) {
      uint otherIdx = (lidU32 + currentStride);
      float myVal = sdata[lidU32];
      float otherVal = sdata[otherIdx];
      sdata[lidU32] = (myVal + otherVal);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lidU32 == 0u) {
    float sumSq = sdata[0u];
    float dist = sqrt(sumSq);
    result[0u] = dist;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_cosine_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_cosine_distance_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* nnz [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdataDot[256];
  threadgroup float sdataNormA[256];
  threadgroup float sdataNormB[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lidU32 = ((uint)(int(__tid.x)));
  uint nnzVal = nnz[0u];
  float localDotAccum = as_type<float>(0x00000000u);
  float localNormASqAccum = as_type<float>(0x00000000u);
  float localNormBSqAccum = as_type<float>(0x00000000u);
  uint idxAccum = gid;
  uint gridSize = ((uint)(int(__grid_size.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      float prod = (aVal * bVal);
      float currentDot = localDotAccum;
      float newDot = (currentDot + prod);
      localDotAccum = newDot;
      float aSq = (aVal * aVal);
      float bSq = (bVal * bVal);
      float currentNormASq = localNormASqAccum;
      float newNormASq = (currentNormASq + aSq);
      localNormASqAccum = newNormASq;
      float currentNormBSq = localNormBSqAccum;
      float newNormBSq = (currentNormBSq + bSq);
      localNormBSqAccum = newNormBSq;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  float finalLocalDot = localDotAccum;
  float finalLocalNormASq = localNormASqAccum;
  float finalLocalNormBSq = localNormBSqAccum;
  sdataDot[lidU32] = finalLocalDot;
  sdataNormA[lidU32] = finalLocalNormASq;
  sdataNormB[lidU32] = finalLocalNormBSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if (lidU32 < currentStride) {
      uint otherIdx = (lidU32 + currentStride);
      float myDot = sdataDot[lidU32];
      float otherDot = sdataDot[otherIdx];
      sdataDot[lidU32] = (myDot + otherDot);
      float myNormA = sdataNormA[lidU32];
      float otherNormA = sdataNormA[otherIdx];
      sdataNormA[lidU32] = (myNormA + otherNormA);
      float myNormB = sdataNormB[lidU32];
      float otherNormB = sdataNormB[otherIdx];
      sdataNormB[lidU32] = (myNormB + otherNormB);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lidU32 == 0u) {
    float dotProduct = sdataDot[0u];
    float normASq = sdataNormA[0u];
    float normBSq = sdataNormB[0u];
    float normA = sqrt(normASq);
    float normB = sqrt(normBSq);
    float normProd = (normA * normB);
    bool normIsZero = (normProd == as_type<float>(0x00000000u));
    float cosineSim = ((normIsZero) ? (as_type<float>(0x00000000u)) : ((dotProduct / normProd)));
    float cosineDist = (as_type<float>(0x3f800000u) - cosineSim);
    float finalResult = ((normIsZero) ? (as_type<float>(0x00000000u)) : (cosineDist));
    result[0u] = finalResult;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_jaccard_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_jaccard_distance_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* nnz [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup uint sdataIntersect[256];
  threadgroup uint sdataUnion[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lidU32 = ((uint)(int(__tid.x)));
  uint nnzVal = nnz[0u];
  uint localIntersectAccum = 0u;
  uint localUnionAccum = 0u;
  uint idxAccum = gid;
  uint gridSize = ((uint)(int(__grid_size.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      bool aNonZero = (aVal != as_type<float>(0x00000000u));
      bool bNonZero = (bVal != as_type<float>(0x00000000u));
      bool inIntersect = (aNonZero && bNonZero);
      bool inUnion = (aNonZero || bNonZero);
      uint intersectInc = ((inIntersect) ? (1u) : (0u));
      uint unionInc = ((inUnion) ? (1u) : (0u));
      uint currentIntersect = localIntersectAccum;
      uint newIntersect = (currentIntersect + intersectInc);
      localIntersectAccum = newIntersect;
      uint currentUnion = localUnionAccum;
      uint newUnion = (currentUnion + unionInc);
      localUnionAccum = newUnion;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  uint finalLocalIntersect = localIntersectAccum;
  uint finalLocalUnion = localUnionAccum;
  sdataIntersect[lidU32] = finalLocalIntersect;
  sdataUnion[lidU32] = finalLocalUnion;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if (lidU32 < currentStride) {
      uint otherIdx = (lidU32 + currentStride);
      uint myIntersect = sdataIntersect[lidU32];
      uint otherIntersect = sdataIntersect[otherIdx];
      sdataIntersect[lidU32] = (myIntersect + otherIntersect);
      uint myUnion = sdataUnion[lidU32];
      uint otherUnion = sdataUnion[otherIdx];
      sdataUnion[lidU32] = (myUnion + otherUnion);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lidU32 == 0u) {
    uint totalIntersect = sdataIntersect[0u];
    uint totalUnion = sdataUnion[0u];
    bool unionIsZero = (totalUnion == 0u);
    float intersectF = ((float)(totalIntersect));
    float unionF = ((float)(totalUnion));
    float jaccardSim = ((unionIsZero) ? (as_type<float>(0x00000000u)) : ((intersectF / unionF)));
    float jaccardDist = (as_type<float>(0x3f800000u) - jaccardSim);
    float finalResult = ((unionIsZero) ? (as_type<float>(0x00000000u)) : (jaccardDist));
    result[0u] = finalResult;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_jacobi_iter_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_jacobi_iter_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* n [[buffer(5)]], device float* x_new [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_type<float>(0x00000000u);
    float diagAccum = as_type<float>(0x3f800000u);
    float bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        bool isDiag = (col == i);
        float currentDiag = diagAccum;
        float newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        float xj = x[col];
        float contrib = (aVal * xj);
        float offDiagContrib = ((isDiag) ? (as_type<float>(0x00000000u)) : (contrib));
        float currentSum = sumAccum;
        float newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    float finalSum = sumAccum;
    float finalDiag = diagAccum;
    float numerator = (bi - finalSum);
    float result = (numerator / finalDiag);
    x_new[i] = result;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_gauss_seidel_iter_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_gauss_seidel_iter_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* n [[buffer(5)]], device float* x_new [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_type<float>(0x00000000u);
    float diagAccum = as_type<float>(0x3f800000u);
    float bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        bool isDiag = (col == i);
        bool isLower = (col < i);
        float currentDiag = diagAccum;
        float newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        float xjNew = x_new[col];
        float xjOld = x[col];
        float xj = ((isLower) ? (xjNew) : (xjOld));
        float contrib = (aVal * xj);
        float offDiagContrib = ((isDiag) ? (as_type<float>(0x00000000u)) : (contrib));
        float currentSum = sumAccum;
        float newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    float finalSum = sumAccum;
    float finalDiag = diagAccum;
    float numerator = (bi - finalSum);
    float result = (numerator / finalDiag);
    x_new[i] = result;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_cg_iter_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_cg_iter_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* p [[buffer(4)]], device const float* r [[buffer(5)]], device float* q [[buffer(6)]], device const float* rTr [[buffer(7)]], device const float* alpha [[buffer(8)]], device const uint* n [[buffer(9)]], device float* x_new [[buffer(10)]], device float* r_new [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    float alphaVal = alpha[0u];
    float xi = x[i];
    float pi = p[i];
    float ri = r[i];
    float qi = q[i];
    float alphaPi = (alphaVal * pi);
    float xNew = (xi + alphaPi);
    x_new[i] = xNew;
    float alphaQi = (alphaVal * qi);
    float rNew = (ri - alphaQi);
    r_new[i] = rNew;
  }
}

)kernel";
const char* k_metal_navatala_sparse_sparse_bicgstab_iter_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_sparse_bicgstab_iter_f32(device const float* x [[buffer(0)]], device const float* p [[buffer(1)]], device const float* r [[buffer(2)]], device const float* v [[buffer(3)]], device float* s [[buffer(4)]], device const float* t [[buffer(5)]], device const float* alpha [[buffer(6)]], device const float* omega [[buffer(7)]], device const uint* n [[buffer(8)]], device float* x_new [[buffer(9)]], device float* r_new [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    float alphaVal = alpha[0u];
    float omegaVal = omega[0u];
    float xi = x[i];
    float pi = p[i];
    float si = s[i];
    float ti = t[i];
    float alphaPi = (alphaVal * pi);
    float omegaSi = (omegaVal * si);
    float xNew = (xi + alphaPi);
    float xNew2 = (xNew + omegaSi);
    x_new[i] = xNew2;
    float omegaTi = (omegaVal * ti);
    float rNew = (si - omegaTi);
    r_new[i] = rNew;
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_residual_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_compute_residual_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* n [[buffer(5)]], device float* residual [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float axSumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        float xj = x[col];
        float contrib = (aVal * xj);
        float currentAxSum = axSumAccum;
        float newAxSum = (currentAxSum + contrib);
        axSumAccum = newAxSum;
      }
    }
    float finalAxSum = axSumAccum;
    float bi = b[i];
    float res = (bi - finalAxSum);
    residual[i] = res;
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_residual_norm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_compute_residual_norm_f32(device const float* residual [[buffer(0)]], device const uint* n [[buffer(1)]], device float* normSq [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint i = gid;
  uint nVal = n[0u];
  bool inBounds = (i < nVal);
  float ri = ((inBounds) ? (residual[i]) : (as_type<float>(0x00000000u)));
  float riSq = (ri * ri);
  sdata[lid] = riSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = 0; s < (int)(128u); ++s) {
    uint sU32 = ((uint)(s));
    if (lid < sU32) {
      float other = sdata[(lid + sU32)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    atomic_fetch_add_explicit((device atomic_float*)(&normSq[0u]), sdata[0u], memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_spmv_row_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_spmv_row_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const uint* n [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        float xj = x[col];
        float term = (aVal * xj);
        float currentSum = sumAccum;
        sumAccum = (currentSum + term);
      }
    }
    y[i] = sumAccum;
  }
}

)kernel";
const char* k_metal_navatala_sparse_chebyshev_smooth_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_chebyshev_smooth_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* xPrev [[buffer(4)]], device const float* diagInv [[buffer(5)]], device const float* b [[buffer(6)]], device const uint* n [[buffer(7)]], device const float* alpha [[buffer(8)]], device const float* beta [[buffer(9)]], device float* xNext [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        float xj = x[col];
        float term = (aVal * xj);
        float currentSum = sumAccum;
        sumAccum = (currentSum + term);
      }
    }
    float bi = b[i];
    float xi = x[i];
    float xPrevI = xPrev[i];
    float dInv = diagInv[i];
    float alphaVal = alpha[0u];
    float betaVal = beta[0u];
    float residual = (bi - sumAccum);
    float scaledResidual = (dInv * residual);
    float history = (xi - xPrevI);
    float update = ((alphaVal * scaledResidual) + (betaVal * history));
    xNext[i] = (xi + update);
  }
}

)kernel";
const char* k_metal_navatala_sparse_find_min_edge_per_component_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline float gpu_atomic_min_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value < cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

static inline float gpu_atomic_max_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value > cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

kernel void navatala_sparse_find_min_edge_per_component_f32(device const uint* srcNodes [[buffer(0)]], device const uint* dstNodes [[buffer(1)]], device const float* weights [[buffer(2)]], device const uint* components [[buffer(3)]], device const uint* numEdges [[buffer(4)]], device const uint* numNodes [[buffer(5)]], device uint* minEdgeIdx [[buffer(6)]], device float* minEdgeWeight [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint edgeIdx = ((uint)(int(__gid.x)));
  if (edgeIdx < numEdges[0u]) {
    uint src = srcNodes[edgeIdx];
    uint dst = dstNodes[edgeIdx];
    uint srcComp = components[src];
    uint dstComp = components[dst];
    if (srcComp != dstComp) {
      float w = weights[edgeIdx];
      float currWeight = minEdgeWeight[srcComp];
      if (w < currWeight) {
        gpu_atomic_min_float((device atomic_float*)(&minEdgeWeight[srcComp]), w);
        minEdgeIdx[srcComp] = edgeIdx;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_merge_components_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_merge_components_f32(device const uint* srcNodes [[buffer(0)]], device const uint* dstNodes [[buffer(1)]], device const uint* minEdgeIdx [[buffer(2)]], device const uint* numNodes [[buffer(3)]], device uint* components [[buffer(4)]], device bool* mstEdges [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint nodeId = ((uint)(int(__gid.x)));
  if (nodeId < numNodes[0u]) {
    uint comp = components[nodeId];
    if (comp == nodeId) {
      uint edgeIdx = minEdgeIdx[nodeId];
      if (edgeIdx != 4294967295u) {
        uint src = srcNodes[edgeIdx];
        uint dst = dstNodes[edgeIdx];
        uint dstComp = components[dst];
        if (nodeId != dstComp) {
          uint smaller = (((nodeId < dstComp)) ? (nodeId) : (dstComp));
          uint larger = (((nodeId < dstComp)) ? (dstComp) : (nodeId));
          atomic_fetch_min_explicit((device atomic_uint*)(&components[larger]), smaller, memory_order_relaxed);
          mstEdges[edgeIdx] = true;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_merge_components_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_merge_components_f64(device const uint* srcNodes [[buffer(0)]], device const uint* dstNodes [[buffer(1)]], device const uint* minEdgeIdx [[buffer(2)]], device const uint* numNodes [[buffer(3)]], device uint* components [[buffer(4)]], device bool* mstEdges [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint nodeId = ((uint)(int(__gid.x)));
  if (nodeId < numNodes[0u]) {
    uint comp = components[nodeId];
    if (comp == nodeId) {
      uint edgeIdx = minEdgeIdx[nodeId];
      if (edgeIdx != 4294967295u) {
        uint src = srcNodes[edgeIdx];
        uint dst = dstNodes[edgeIdx];
        uint dstComp = components[dst];
        if (nodeId != dstComp) {
          uint smaller = (((nodeId < dstComp)) ? (nodeId) : (dstComp));
          uint larger = (((nodeId < dstComp)) ? (dstComp) : (nodeId));
          atomic_fetch_min_explicit((device atomic_uint*)(&components[larger]), smaller, memory_order_relaxed);
          mstEdges[edgeIdx] = true;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_mst_boruvka_step_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline float gpu_atomic_min_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value < cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

static inline float gpu_atomic_max_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value > cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

kernel void navatala_sparse_mst_boruvka_step_f32(device const uint* srcNodes [[buffer(0)]], device const uint* dstNodes [[buffer(1)]], device const float* weights [[buffer(2)]], device const uint* numEdges [[buffer(3)]], device const uint* numNodes [[buffer(4)]], device uint* components [[buffer(5)]], device uint* minEdgeIdx [[buffer(6)]], device float* minEdgeWeight [[buffer(7)]], device bool* mstEdges [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  if (tid < numEdges[0u]) {
    uint src = srcNodes[tid];
    uint dst = dstNodes[tid];
    uint srcRoot = components[src];
    uint dstRoot = components[dst];
    if (srcRoot != dstRoot) {
      float w = weights[tid];
      float currMin = minEdgeWeight[srcRoot];
      if (w < currMin) {
        gpu_atomic_min_float((device atomic_float*)(&minEdgeWeight[srcRoot]), w);
        minEdgeIdx[srcRoot] = tid;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_sp_m_v_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_sp_m_v_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const uint* nRows [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rowStart = ((int)(rowPtr[row]));
    int rowEnd = ((int)(rowPtr[(row + 1)]));
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      int k = (rowStart + j);
      int col = ((int)(colIdx[k]));
      float aij = values[k];
      float xj = x[col];
      acc = (acc + (aij * xj));
    }
    y[row] = acc;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_sp_m_v_scaled_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_sp_m_v_scaled_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* alpha [[buffer(4)]], device const float* beta [[buffer(5)]], device const uint* nRows [[buffer(6)]], device float* y [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rowStart = ((int)(rowPtr[row]));
    int rowEnd = ((int)(rowPtr[(row + 1)]));
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      int k = (rowStart + j);
      int col = ((int)(colIdx[k]));
      float aij = values[k];
      float xj = x[col];
      acc = (acc + (aij * xj));
    }
    float alphaVal = alpha[0];
    float betaVal = beta[0];
    float yi = y[row];
    y[row] = ((alphaVal * acc) + (betaVal * yi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_residual_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_residual_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* nRows [[buffer(5)]], device float* r [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rowStart = ((int)(rowPtr[row]));
    int rowEnd = ((int)(rowPtr[(row + 1)]));
    float Ax = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      int k = (rowStart + j);
      int col = ((int)(colIdx[k]));
      float aij = values[k];
      float xj = x[col];
      Ax = (Ax + (aij * xj));
    }
    float bi = b[row];
    r[row] = (bi - Ax);
  }
}

)kernel";
const char* k_metal_navatala_sparse_axpy_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_axpy_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device const uint* n [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float xi = x[gid];
    float yi = y[gid];
    y[gid] = (yi + (a * xi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_xpay_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_xpay_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device const uint* n [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float xi = x[gid];
    float yi = y[gid];
    y[gid] = (xi + (a * yi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_dot_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dot_partials_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n [[buffer(2)]], device float* partials [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float prod = (((gid < N)) ? ((x[gid] * y[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(prod);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_scal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_scal_f32(device const float* alpha [[buffer(0)]], device const uint* n [[buffer(1)]], device float* x [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float xi = x[gid];
    x[gid] = (a * xi);
  }
}

)kernel";
const char* k_metal_navatala_sparse_nrm2_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_nrm2_partials_f32(device const float* x [[buffer(0)]], device const uint* n [[buffer(1)]], device float* partials [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((x[gid] * x[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_zero_fill_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_zero_fill_f32(device const uint* n [[buffer(0)]], device float* x [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    x[gid] = as_type<float>(0x00000000u);
  }
}

)kernel";
const char* k_metal_navatala_sparse_zero_fill_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_zero_fill_u32(device const uint* n [[buffer(0)]], device uint* x [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    x[gid] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_sparse_zero_scalar_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_zero_scalar_u32(device uint* x [[buffer(0)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    x[0] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_sparse_fill_neg_one_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_fill_neg_one_i32(device const uint* n [[buffer(0)]], device int* x [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    x[gid] = -1;
  }
}

)kernel";
const char* k_metal_navatala_sparse_copy_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_copy_f32(device const float* x [[buffer(0)]], device const uint* n [[buffer(1)]], device float* y [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    y[gid] = x[gid];
  }
}

)kernel";
const char* k_metal_navatala_sparse_bsr_sp_m_v_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bsr_sp_m_v_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const uint* nBlockRows [[buffer(4)]], device const uint* blockDim [[buffer(5)]], device float* y [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int blockRow = int(__gid.x);
  int N = ((int)(nBlockRows[0]));
  int bd = ((int)(blockDim[0]));
  if (blockRow < N) {
    int rs = ((int)(rowPtr[blockRow]));
    int re = ((int)(rowPtr[(blockRow + 1)]));
    for (int comp = 0; comp < (int)(bd); ++comp) {
      float sum = as_type<float>(0x00000000u);
      for (int jj = 0; jj < (int)((re - rs)); ++jj) {
        int blk = (rs + jj);
        int blkCol = ((int)(colIdx[blk]));
        for (int c2 = 0; c2 < (int)(bd); ++c2) {
          float a = values[(((((blk * bd) * bd) + (comp * bd)) + c2) + 0)];
          float xval = x[((blkCol * bd) + c2)];
          sum = (sum + (a * xval));
        }
      }
      y[((blockRow * bd) + comp)] = sum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_transpose_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_csr_transpose_count_f32(device const uint* colIdx [[buffer(0)]], device const uint* nnz [[buffer(1)]], device const uint* nCols [[buffer(2)]], device atomic_uint* TrowCount [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnz[0]));
  if (gid < N) {
    int col = ((int)(colIdx[gid]));
    uint _aod1 = atomic_fetch_add_explicit(((device atomic_uint*)(&(TrowCount[col]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_transpose_fill_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_csr_transpose_fill_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* TrowPtr [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* TcolIdx [[buffer(5)]], device float* Tvalues [[buffer(6)]], device atomic_uint* TwritePos [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float val = values[k];
      uint pos = atomic_fetch_add_explicit(((device atomic_uint*)(&(TwritePos[col]))), 1u, memory_order_relaxed);
      int tp = ((int)(TrowPtr[col]));
      int tgt = (tp + ((int)(pos)));
      TcolIdx[tgt] = ((uint)(gid));
      Tvalues[tgt] = val;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_row_norm_l1_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_row_norm_l1_f32(device const uint* rowPtr [[buffer(0)]], device const float* values [[buffer(1)]], device const uint* nRows [[buffer(2)]], device float* rowNorm [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float a = values[k];
      sum = (sum + abs(a));
    }
    rowNorm[gid] = sum;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_row_norm_linf_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_row_norm_linf_f32(device const uint* rowPtr [[buffer(0)]], device const float* values [[buffer(1)]], device const uint* nRows [[buffer(2)]], device float* rowNorm [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float mx = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float a = values[k];
      mx = (((abs(a) > mx)) ? (abs(a)) : (mx));
    }
    rowNorm[gid] = mx;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_frobenius_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_frobenius_partials_f32(device const float* values [[buffer(0)]], device const uint* nnz [[buffer(1)]], device float* partials [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnz[0]));
  if (gid < N) {
    float a = values[gid];
    partials[gid] = (a * a);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_truncate_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_truncate_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* tol [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* newRowNnz [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float rowMax = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float a = values[k];
      rowMax = (((abs(a) > rowMax)) ? (abs(a)) : (rowMax));
    }
    float tolVal = tol[0];
    float thresh = (tolVal * rowMax);
    int count = 0;
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float a2 = values[k2];
      count = (count + (((abs(a2) >= thresh)) ? (1) : (0)));
    }
    newRowNnz[gid] = ((uint)(count));
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_compact_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_compact_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* newRowPtr [[buffer(3)]], device const uint* keepMask [[buffer(4)]], device const uint* nRows [[buffer(5)]], device uint* newColIdx [[buffer(6)]], device float* newValues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int newRs = ((int)(newRowPtr[gid]));
    int pos = 0;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint keep = keepMask[k];
      if (keep == 1u) {
        int tgt = (newRs + pos);
        uint c = colIdx[k];
        float v = values[k];
        newColIdx[tgt] = c;
        newValues[tgt] = v;
        pos = (pos + 1);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_sort_columns_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_sort_columns_f32(device const uint* rowPtr [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* colIdx [[buffer(2)]], device float* values [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int len = (re - rs);
    for (int i = 0; i < (int)((len - 1)); ++i) {
      for (int j = 0; j < (int)(((len - 1) - i)); ++j) {
        int a = (rs + j);
        int b = (rs + (j + 1));
        uint ca = colIdx[a];
        uint cb = colIdx[b];
        if (ca > cb) {
          colIdx[a] = cb;
          colIdx[b] = ca;
          float va = values[a];
          float vb = values[b];
          values[a] = vb;
          values[b] = va;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_permute_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_permute_rows_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* perm [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* newRowPtr [[buffer(5)]], device uint* newColIdx [[buffer(6)]], device float* newValues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int origRow = ((int)(perm[gid]));
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int nnz = (re - rs);
    newRowPtr[gid] = ((uint)(nnz));
    for (int j = 0; j < (int)(nnz); ++j) {
      int k = (rs + j);
      uint c = colIdx[k];
      float v = values[k];
      newColIdx[(gid + j)] = c;
      newValues[(gid + j)] = v;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_check_diag_dominance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_csr_check_diag_dominance_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device uint* isDiagDom [[buffer(4)]], device atomic_uint* allDiagDom [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float diag = as_type<float>(0x00000000u);
    float offSum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      if (col == gid) {
        diag = abs(a);
      } else {
        offSum = (offSum + abs(a));
      }
    }
    uint ok = (((diag >= offSum)) ? (1u) : (0u));
    isDiagDom[gid] = ok;
    uint _aod2 = atomic_fetch_and_explicit(((device atomic_uint*)(&(allDiagDom[0]))), ok, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_row_nnz_histogram = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_csr_row_nnz_histogram(device const uint* rowPtr [[buffer(0)]], device const uint* nRows [[buffer(1)]], device const uint* maxBins [[buffer(2)]], device atomic_uint* histogram [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nnz = (re - rs);
    int maxB = ((int)(maxBins[0]));
    if (nnz < maxB) {
      uint _aod3 = atomic_fetch_add_explicit(((device atomic_uint*)(&(histogram[nnz]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_dense_l_u_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dense_l_u_f32(device const uint* n [[buffer(0)]], device float* A [[buffer(1)]], device uint* pivots [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int N = ((int)(n[0]));
  for (int k = 0; k < (int)(N); ++k) {
    float maxVal = A[((k * N) + k)];
    int maxRow = k;
    for (int i = 0; i < (int)((N - k)); ++i) {
      int r = (k + i);
      float v = A[((r * N) + k)];
      if (abs(v) > abs(maxVal)) {
        maxVal = v;
        maxRow = r;
      }
    }
    pivots[k] = ((uint)(maxRow));
    if (maxRow != k) {
      for (int j = 0; j < (int)(N); ++j) {
        float a = A[((k * N) + j)];
        float b = A[((maxRow * N) + j)];
        A[((k * N) + j)] = b;
        A[((maxRow * N) + j)] = a;
      }
    }
    float pivot = A[((k * N) + k)];
    for (int i2 = 0; i2 < (int)((N - (k + 1))); ++i2) {
      int r2 = ((k + 1) + i2);
      float factor = (A[((r2 * N) + k)] / pivot);
      A[((r2 * N) + k)] = factor;
      for (int j2 = 0; j2 < (int)((N - (k + 1))); ++j2) {
        int c = ((k + 1) + j2);
        float old = A[((r2 * N) + c)];
        A[((r2 * N) + c)] = (old - (factor * A[((k * N) + c)]));
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_dense_l_u_solve_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dense_l_u_solve_f32(device const float* LU [[buffer(0)]], device const uint* pivots [[buffer(1)]], device const float* b [[buffer(2)]], device const uint* n [[buffer(3)]], device float* x [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int N = ((int)(n[0]));
  for (int iCpy = 0; iCpy < (int)(N); ++iCpy) {
    x[iCpy] = b[iCpy];
  }
  for (int k = 0; k < (int)(N); ++k) {
    int pk = ((int)(pivots[k]));
    if (pk != k) {
      float tmp = x[k];
      x[k] = x[pk];
      x[pk] = tmp;
    }
  }
  for (int i2 = 0; i2 < (int)(N); ++i2) {
    float sum = x[i2];
    for (int j = 0; j < (int)(i2); ++j) {
      float lij = LU[((i2 * N) + j)];
      sum = (sum - (lij * x[j]));
    }
    x[i2] = sum;
  }
  for (int ii = 0; ii < (int)(N); ++ii) {
    int i3 = ((N - 1) - ii);
    float sum2 = x[i3];
    int start = (i3 + 1);
    for (int j2 = 0; j2 < (int)((N - start)); ++j2) {
      int c = (start + j2);
      float uij = LU[((i3 * N) + c)];
      sum2 = (sum2 - (uij * x[c]));
    }
    float diag = LU[((i3 * N) + i3)];
    x[i3] = (sum2 / diag);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_replace_values_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_replace_values_f32(device const float* newValues [[buffer(0)]], device const uint* nnz [[buffer(1)]], device float* values [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnz[0]));
  if (gid < N) {
    float v = newValues[gid];
    values[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_to_coo_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_to_coo_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device uint* rowIdx [[buffer(4)]], device uint* cooColIdx [[buffer(5)]], device float* cooValues [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      rowIdx[k] = ((uint)(gid));
      cooColIdx[k] = colIdx[k];
      cooValues[k] = values[k];
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_coo_to_csr_row_count = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_coo_to_csr_row_count(device const uint* rowIdx [[buffer(0)]], device const uint* nnz [[buffer(1)]], device const uint* nRows [[buffer(2)]], device atomic_uint* rowCount [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnz[0]));
  if (gid < N) {
    int row = ((int)(rowIdx[gid]));
    uint _aod4 = atomic_fetch_add_explicit(((device atomic_uint*)(&(rowCount[row]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_to_ell_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_to_ell_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* maxNnzPerRow [[buffer(4)]], device uint* ellColIdx [[buffer(5)]], device float* ellValues [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nnz = (re - rs);
    int maxNnz = ((int)(maxNnzPerRow[0]));
    for (int j = 0; j < (int)(maxNnz); ++j) {
      if (j < nnz) {
        int k = (rs + j);
        ellColIdx[((gid * maxNnz) + j)] = colIdx[k];
        ellValues[((gid * maxNnz) + j)] = values[k];
      } else {
        ellColIdx[((gid * maxNnz) + j)] = 4294967295u;
        ellValues[((gid * maxNnz) + j)] = as_type<float>(0x00000000u);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_ell_sp_m_v_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ell_sp_m_v_f32(device const uint* ellColIdx [[buffer(0)]], device const float* ellValues [[buffer(1)]], device const float* x [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* maxNnzPerRow [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int maxNnz = ((int)(maxNnzPerRow[0]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(maxNnz); ++j) {
      int ellIdx = ((gid * maxNnz) + j);
      uint col = ellColIdx[ellIdx];
      if (col < 4294967295u) {
        float a = ellValues[ellIdx];
        float xj = x[((int)(col))];
        sum = (sum + (a * xj));
      }
    }
    y[gid] = sum;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_add_symbolic_row_nnz = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_add_symbolic_row_nnz(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const uint* BrowPtr [[buffer(2)]], device const uint* BcolIdx [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* CrowNnz [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int brs = ((int)(BrowPtr[gid]));
    int bre = ((int)(BrowPtr[(gid + 1)]));
    int annz = (are - ars);
    int bnnz = (bre - brs);
    int count = 0;
    int ai = 0;
    int bi = 0;
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!((ai < annz) && (bi < bnnz))) break;
      int acol = ((int)(AcolIdx[(ars + ai)]));
      int bcol = ((int)(BcolIdx[(brs + bi)]));
      if (acol == bcol) {
        count = (count + 1);
        ai = (ai + 1);
        bi = (bi + 1);
      } else {
        if (acol < bcol) {
          count = (count + 1);
          ai = (ai + 1);
        } else {
          count = (count + 1);
          bi = (bi + 1);
        }
      }
    }
    count = (count + (annz - ai));
    count = (count + (bnnz - bi));
    CrowNnz[gid] = ((uint)(count));
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_add_numeric_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_add_numeric_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const uint* BrowPtr [[buffer(3)]], device const uint* BcolIdx [[buffer(4)]], device const float* Bvalues [[buffer(5)]], device const uint* CrowPtr [[buffer(6)]], device const float* alpha [[buffer(7)]], device const float* beta [[buffer(8)]], device const uint* nRows [[buffer(9)]], device uint* CcolIdx [[buffer(10)]], device float* Cvalues [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int brs = ((int)(BrowPtr[gid]));
    int bre = ((int)(BrowPtr[(gid + 1)]));
    int crs = ((int)(CrowPtr[gid]));
    float a = alpha[0];
    float b = beta[0];
    int ci = 0;
    int ai = 0;
    int bi = 0;
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!((ai < (are - ars)) && (bi < (bre - brs)))) break;
      int acol = ((int)(AcolIdx[(ars + ai)]));
      int bcol = ((int)(BcolIdx[(brs + bi)]));
      if (acol == bcol) {
        int tgt = (crs + ci);
        CcolIdx[tgt] = ((uint)(acol));
        Cvalues[tgt] = ((a * Avalues[(ars + ai)]) + (b * Bvalues[(brs + bi)]));
        ci = (ci + 1);
        ai = (ai + 1);
        bi = (bi + 1);
      } else {
        if (acol < bcol) {
          int tgt2 = (crs + ci);
          CcolIdx[tgt2] = ((uint)(acol));
          Cvalues[tgt2] = (a * Avalues[(ars + ai)]);
          ci = (ci + 1);
          ai = (ai + 1);
        } else {
          int tgt3 = (crs + ci);
          CcolIdx[tgt3] = ((uint)(bcol));
          Cvalues[tgt3] = (b * Bvalues[(brs + bi)]);
          ci = (ci + 1);
          bi = (bi + 1);
        }
      }
    }
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!(ai < (are - ars))) break;
      int tgt4 = (crs + ci);
      int acol2 = ((int)(AcolIdx[(ars + ai)]));
      CcolIdx[tgt4] = ((uint)(acol2));
      Cvalues[tgt4] = (a * Avalues[(ars + ai)]);
      ci = (ci + 1);
      ai = (ai + 1);
    }
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!(bi < (bre - brs))) break;
      int tgt5 = (crs + ci);
      int bcol2 = ((int)(BcolIdx[(brs + bi)]));
      CcolIdx[tgt5] = ((uint)(bcol2));
      Cvalues[tgt5] = (b * Bvalues[(brs + bi)]);
      ci = (ci + 1);
      bi = (bi + 1);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_scale_f32(device const float* alpha [[buffer(0)]], device const uint* nnz [[buffer(1)]], device float* values [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnz[0]));
  if (gid < N) {
    float v = values[gid];
    float a = alpha[0];
    values[gid] = (a * v);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_symmetry_check_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_csr_symmetry_check_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* tol [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* asymmetricFlags [[buffer(5)]], device atomic_uint* isSymmetric [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col < gid) {
      } else {
        if (col > gid) {
          float a = values[k];
          int colRs = ((int)(rowPtr[col]));
          int colRe = ((int)(rowPtr[(col + 1)]));
          uint found = 0u;
          for (int jj = 0; jj < (int)((colRe - colRs)); ++jj) {
            int kk = (colRs + jj);
            int cc = ((int)(colIdx[kk]));
            if (cc == gid) {
              float at = values[kk];
              float tolVal = tol[0];
              float diff = abs((a - at));
              if (diff > tolVal) {
                asymmetricFlags[gid] = 1u;
                uint _aod5 = atomic_fetch_and_explicit(((device atomic_uint*)(&(isSymmetric[0]))), 0u, memory_order_relaxed);
              }
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_nnz_per_row = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_nnz_per_row(device const uint* rowPtr [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* nnzPerRow [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    uint rs = rowPtr[gid];
    uint re = rowPtr[(gid + 1)];
    nnzPerRow[gid] = (re - rs);
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_max_nnz_per_row_partials = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_max_nnz_per_row_partials(device const uint* nnzPerRow [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* partials [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    partials[gid] = nnzPerRow[gid];
  } else {
    partials[gid] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_extract_diag_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_extract_diag_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device float* diag [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float d = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col == gid) {
        d = values[k];
      }
    }
    diag[gid] = d;
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_permute_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_permute_cols_f32(device const uint* colIdx [[buffer(0)]], device const float* values [[buffer(1)]], device const uint* perm [[buffer(2)]], device const uint* nnz [[buffer(3)]], device uint* newColIdx [[buffer(4)]], device float* newValues [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnz[0]));
  if (gid < N) {
    uint c = colIdx[gid];
    uint newC = perm[((int)(c))];
    newColIdx[gid] = newC;
    newValues[gid] = values[gid];
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_row_slice_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_row_slice_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* rowBegin [[buffer(3)]], device const uint* rowEnd [[buffer(4)]], device uint* subRowPtr [[buffer(5)]], device uint* subColIdx [[buffer(6)]], device float* subValues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int begin = ((int)(rowBegin[0]));
  int end = ((int)(rowEnd[0]));
  int nRows = (end - begin);
  if (gid < nRows) {
    int origRow = (begin + gid);
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int subRs = ((int)(rowPtr[begin]));
    subRowPtr[gid] = ((uint)((rs - subRs)));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      subColIdx[k] = colIdx[k];
      subValues[k] = values[k];
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cg_update_x_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_update_x_r_f32(device const float* p [[buffer(0)]], device const float* Ap [[buffer(1)]], device const float* alpha [[buffer(2)]], device const uint* n [[buffer(3)]], device float* x [[buffer(4)]], device float* r [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float pi = p[gid];
    float api = Ap[gid];
    float xi = x[gid];
    float ri = r[gid];
    x[gid] = (xi + (a * pi));
    r[gid] = (ri - (a * api));
  }
}

)kernel";
const char* k_metal_navatala_sparse_cg_update_p_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_update_p_f32(device const float* z [[buffer(0)]], device const float* beta [[buffer(1)]], device const uint* n [[buffer(2)]], device float* p [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float b = beta[0];
    float zi = z[gid];
    float pi = p[gid];
    p[gid] = (zi + (b * pi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_cg_compute_alpha_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_compute_alpha_f32(device const float* rTz [[buffer(0)]], device const float* pTAp [[buffer(1)]], device float* alpha [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float rz = rTz[0];
  float pa = pTAp[0];
  float result = (rz / pa);
  alpha[0] = result;
}

)kernel";
const char* k_metal_navatala_sparse_cg_compute_beta_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_compute_beta_f32(device const float* rTzNew [[buffer(0)]], device const float* rTzOld [[buffer(1)]], device float* beta [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float rzNew = rTzNew[0];
  float rzOld = rTzOld[0];
  float result = (rzNew / rzOld);
  beta[0] = result;
}

)kernel";
const char* k_metal_navatala_sparse_cg_fused_update_dot_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_fused_update_dot_f32(device const float* p [[buffer(0)]], device const float* Ap [[buffer(1)]], device const float* z [[buffer(2)]], device const float* alpha [[buffer(3)]], device const uint* n [[buffer(4)]], device float* x [[buffer(5)]], device float* r [[buffer(6)]], device float* rTzPartials [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float a = alpha[0];
  float pi = (((gid < N)) ? (p[gid]) : (as_type<float>(0x00000000u)));
  float api = (((gid < N)) ? (Ap[gid]) : (as_type<float>(0x00000000u)));
  float zi = (((gid < N)) ? (z[gid]) : (as_type<float>(0x00000000u)));
  if (gid < N) {
    float xi = x[gid];
    float ri = r[gid];
    float xnew = (xi + (a * pi));
    float rnew = (ri - (a * api));
    x[gid] = xnew;
    r[gid] = rnew;
  }
  float rz = ((r[gid] - (a * api)) * zi);
  float warpSum = simd_sum(rz);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      rTzPartials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cg_init_residual_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_init_residual_f32(device const float* b [[buffer(0)]], device const float* Ax [[buffer(1)]], device const uint* n [[buffer(2)]], device float* r [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float bi = b[gid];
    float ai = Ax[gid];
    r[gid] = (bi - ai);
  }
}

)kernel";
const char* k_metal_navatala_sparse_cg_residual_norm_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cg_residual_norm_partials_f32(device const float* r [[buffer(0)]], device const uint* n [[buffer(1)]], device float* partials [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((r[gid] * r[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_update_p_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_update_p_f32(device const float* r [[buffer(0)]], device const float* v [[buffer(1)]], device const float* beta [[buffer(2)]], device const float* omega [[buffer(3)]], device const uint* n [[buffer(4)]], device float* p [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float ri = r[gid];
    float pi = p[gid];
    float vi = v[gid];
    float b = beta[0];
    float w = omega[0];
    float result = (ri + (b * (pi - (w * vi))));
    p[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_update_x_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_update_x_f32(device const float* phat [[buffer(0)]], device const float* shat [[buffer(1)]], device const float* alpha [[buffer(2)]], device const float* omega [[buffer(3)]], device const uint* n [[buffer(4)]], device float* x [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float xi = x[gid];
    float ph = phat[gid];
    float sh = shat[gid];
    float a = alpha[0];
    float w = omega[0];
    float result = (xi + ((a * ph) + (w * sh)));
    x[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_compute_beta_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_compute_beta_f32(device const float* rhoNew [[buffer(0)]], device const float* rhoOld [[buffer(1)]], device const float* alpha [[buffer(2)]], device const float* omega [[buffer(3)]], device float* beta [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float rn = rhoNew[0];
  float ro = rhoOld[0];
  float a = alpha[0];
  float w = omega[0];
  float result = ((rn / ro) * (a / w));
  beta[0] = result;
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_compute_omega_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_compute_omega_f32(device const float* dotTS [[buffer(0)]], device const float* dotTT [[buffer(1)]], device float* omega [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float ts = dotTS[0];
  float tt = dotTT[0];
  float result = (ts / tt);
  omega[0] = result;
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_update_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_update_r_f32(device const float* s [[buffer(0)]], device const float* t [[buffer(1)]], device const float* omega [[buffer(2)]], device const uint* n [[buffer(3)]], device float* r [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float om = omega[0];
    float si = s[gid];
    float ti = t[gid];
    r[gid] = (si - (om * ti));
  }
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_compute_s_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_compute_s_f32(device const float* r [[buffer(0)]], device const float* v [[buffer(1)]], device const float* alpha [[buffer(2)]], device const uint* n [[buffer(3)]], device float* s [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float ri = r[gid];
    float vi = v[gid];
    s[gid] = (ri - (a * vi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_compute_alpha_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_compute_alpha_f32(device const float* rho [[buffer(0)]], device const float* r0hatV [[buffer(1)]], device float* alpha [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float r = rho[0];
  float rv = r0hatV[0];
  float result = (r / rv);
  alpha[0] = result;
}

)kernel";
const char* k_metal_navatala_sparse_bicgstab_fused_update_all_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bicgstab_fused_update_all_f32(device const float* phat [[buffer(0)]], device const float* shat [[buffer(1)]], device const float* s [[buffer(2)]], device const float* t [[buffer(3)]], device const float* alpha [[buffer(4)]], device const float* omega [[buffer(5)]], device const uint* n [[buffer(6)]], device float* x [[buffer(7)]], device float* r [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float om = omega[0];
    float ph = phat[gid];
    float sh = shat[gid];
    float si = s[gid];
    float ti = t[gid];
    float xi = x[gid];
    x[gid] = (xi + ((a * ph) + (om * sh)));
    r[gid] = (si - (om * ti));
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_arnoldi_project_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_arnoldi_project_f32(device const float* v [[buffer(0)]], device const float* coeff [[buffer(1)]], device const uint* n [[buffer(2)]], device float* w [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float c = coeff[0];
    float vi = v[gid];
    float wi = w[gid];
    w[gid] = (wi - (c * vi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_normalize_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_normalize_f32(device const float* w [[buffer(0)]], device const float* norm [[buffer(1)]], device const uint* n [[buffer(2)]], device float* v [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float nm = norm[0];
    float wi = w[gid];
    v[gid] = (wi / nm);
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_update_solution_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_update_solution_f32(device const float* V [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device float* x [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int M = ((int)(m[0]));
  if (gid < N) {
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(M); ++j) {
      float yj = y[j];
      float vij = V[(gid + (j * N))];
      acc = (acc + (yj * vij));
    }
    float xi = x[gid];
    x[gid] = (xi + acc);
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_givens_rotation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_givens_rotation_f32(device const float* cs [[buffer(0)]], device const float* sn [[buffer(1)]], device const uint* colIdx [[buffer(2)]], device float* h [[buffer(3)]], device float* csNew [[buffer(4)]], device float* snNew [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int ci = ((int)(colIdx[0]));
  for (int k = 0; k < (int)(ci); ++k) {
    float csk = cs[k];
    float snk = sn[k];
    float hk = h[k];
    float hk1 = h[(k + 1)];
    h[k] = ((csk * hk) + (snk * hk1));
    h[(k + 1)] = ((csk * hk1) - (snk * hk));
  }
  float hj = h[ci];
  float hj1 = h[(ci + 1)];
  float r = sqrt(((hj * hj) + (hj1 * hj1)));
  float csVal = (hj / r);
  float snVal = (hj1 / r);
  csNew[0] = csVal;
  snNew[0] = snVal;
  h[ci] = r;
  h[(ci + 1)] = as_type<float>(0x00000000u);
}

)kernel";
const char* k_metal_navatala_sparse_gmres_arnoldi_batch_project_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_arnoldi_batch_project_f32(device const float* V [[buffer(0)]], device const float* hCol [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* j [[buffer(3)]], device float* w [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int J = ((int)(j[0]));
  if (gid < N) {
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(J); ++k) {
      float hk = hCol[k];
      float vik = V[(gid + (k * N))];
      acc = (acc + (hk * vik));
    }
    float wi = w[gid];
    w[gid] = (wi - acc);
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_hessenberg_dot_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_hessenberg_dot_partials_f32(device const float* V [[buffer(0)]], device const float* w [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* j [[buffer(3)]], device float* partials [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((w[gid] * w[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_restart_init_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_restart_init_f32(device const float* r [[buffer(0)]], device const float* rNorm [[buffer(1)]], device const uint* n [[buffer(2)]], device float* V [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float nm = rNorm[0];
    float ri = r[gid];
    V[gid] = (ri / nm);
  }
}

)kernel";
const char* k_metal_navatala_sparse_gmres_back_solve_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_back_solve_f32(device const float* H [[buffer(0)]], device const float* g [[buffer(1)]], device const uint* m [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int M = ((int)(m[0]));
  int i = (M - 1);
  float gi = g[i];
  float hii = H[((i * M) + i)];
  y[i] = (gi / hii);
}

)kernel";
const char* k_metal_navatala_sparse_gmres_store_basis_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_gmres_store_basis_f32(device const float* w [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* colIdx [[buffer(2)]], device float* V [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int col = ((int)(colIdx[0]));
  if (gid < N) {
    float wi = w[gid];
    V[(gid + (col * N))] = wi;
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_shadow_project_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_shadow_project_f32(device const float* P [[buffer(0)]], device const float* r [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* s [[buffer(3)]], device float* c [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int k = int(__tgid.x);
  int N = ((int)(n[0]));
  float val = (((gid < N)) ? ((P[(gid + (k * N))] * r[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(val);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float sv = sdata[lid];
    float finalSum = simd_sum(sv);
    if (lid == 0) {
      c[k] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_update_x_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_update_x_r_f32(device const float* t [[buffer(0)]], device const float* v [[buffer(1)]], device const float* omega [[buffer(2)]], device const uint* n [[buffer(3)]], device float* x [[buffer(4)]], device float* r [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float om = omega[0];
    float ti = t[gid];
    float vi = v[gid];
    float xi = x[gid];
    float ri = r[gid];
    x[gid] = (xi + (om * ti));
    r[gid] = (ri - (om * vi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_g_space_update_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_g_space_update_f32(device const float* dRj [[buffer(0)]], device const float* alpha [[buffer(1)]], device const uint* n [[buffer(2)]], device float* dRk [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float dj = dRj[gid];
    float dk = dRk[gid];
    dRk[gid] = (dk - (a * dj));
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_compute_omega_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_compute_omega_f32(device const float* dotTS [[buffer(0)]], device const float* dotTT [[buffer(1)]], device float* omega [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float ts = dotTS[0];
  float tt = dotTT[0];
  float result = (ts / tt);
  omega[0] = result;
}

)kernel";
const char* k_metal_navatala_sparse_idr_build_shadow_space_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_build_shadow_space_f32(device const uint* n [[buffer(0)]], device const uint* s [[buffer(1)]], device const uint* seed [[buffer(2)]], device float* P [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  int total = (N * S);
  if (gid < total) {
    int row = (gid % N);
    int col = (gid / N);
    float val = (((row == col)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    P[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_m_sync_fused_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_m_sync_fused_f32(device const float* P [[buffer(0)]], device const float* r [[buffer(1)]], device const float* Minv [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* s [[buffer(4)]], device float* c [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int k = int(__tgid.x);
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  float acc = as_type<float>(0x00000000u);
  for (int j = 0; j < (int)(S); ++j) {
    float pjr = (((gid < N)) ? ((P[(gid + (j * N))] * r[gid])) : (as_type<float>(0x00000000u)));
    float mij = Minv[(k + (j * S))];
    acc = (acc + (mij * pjr));
  }
  float warpSum = simd_sum(acc);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float sv = sdata[lid];
    float finalSum = simd_sum(sv);
    if (lid == 0) {
      c[k] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_compute_v_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_compute_v_f32(device const float* r [[buffer(0)]], device const float* dR [[buffer(1)]], device const float* c [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* s [[buffer(4)]], device float* v [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  if (gid < N) {
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(S); ++k) {
      float ck = c[k];
      float dr = dR[(gid + (k * N))];
      acc = (acc + (ck * dr));
    }
    float ri = r[gid];
    v[gid] = (ri - acc);
  }
}

)kernel";
const char* k_metal_navatala_sparse_idr_store_d_r_column_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_idr_store_d_r_column_f32(device const float* rNew [[buffer(0)]], device const float* rOld [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* colIdx [[buffer(3)]], device float* dR [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int col = ((int)(colIdx[0]));
  if (gid < N) {
    float rn = rNew[gid];
    float ro = rOld[gid];
    dR[(gid + (col * N))] = (rn - ro);
  }
}

)kernel";
const char* k_metal_navatala_sparse_pcg_update_x_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_pcg_update_x_r_f32(device const float* z [[buffer(0)]], device const float* q [[buffer(1)]], device const float* alpha [[buffer(2)]], device const uint* n [[buffer(3)]], device float* x [[buffer(4)]], device float* r [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float zi = z[gid];
    float qi = q[gid];
    float xi = x[gid];
    float ri = r[gid];
    x[gid] = (xi + (a * zi));
    r[gid] = (ri - (a * qi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_pcgf_update_p_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_pcgf_update_p_f32(device const float* z [[buffer(0)]], device const float* zPrev [[buffer(1)]], device const float* r [[buffer(2)]], device const float* beta [[buffer(3)]], device const uint* n [[buffer(4)]], device float* p [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float b = beta[0];
    float zi = z[gid];
    float pi = p[gid];
    p[gid] = (zi + (b * pi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_pbicgstab_fused_s_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_pbicgstab_fused_s_f32(device const float* r [[buffer(0)]], device const float* v [[buffer(1)]], device const float* alpha [[buffer(2)]], device const uint* n [[buffer(3)]], device float* s [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float ri = r[gid];
    float vi = v[gid];
    s[gid] = (ri - (a * vi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_pbicgstab_triple_dot_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_pbicgstab_triple_dot_partials_f32(device const float* r0hat [[buffer(0)]], device const float* r [[buffer(1)]], device const float* v [[buffer(2)]], device const float* s [[buffer(3)]], device const uint* n [[buffer(4)]], device float* partials [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  float h0 = ((inBounds) ? (r0hat[gid]) : (as_type<float>(0x00000000u)));
  float d0 = ((inBounds) ? ((h0 * r[gid])) : (as_type<float>(0x00000000u)));
  float d1 = ((inBounds) ? ((h0 * v[gid])) : (as_type<float>(0x00000000u)));
  float d2 = ((inBounds) ? ((h0 * s[gid])) : (as_type<float>(0x00000000u)));
  float ws0 = simd_sum(d0);
  float ws1 = simd_sum(d1);
  float ws2 = simd_sum(d2);
  threadgroup float sd0[32];
  threadgroup float sd1[32];
  threadgroup float sd2[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sd0[warpIdx] = ws0;
    sd1[warpIdx] = ws1;
    sd2[warpIdx] = ws2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    int grpId = int(__tgid.x);
    int base = (grpId * 3);
    float f0 = simd_sum(sd0[lid]);
    float f1 = simd_sum(sd1[lid]);
    float f2 = simd_sum(sd2[lid]);
    if (lid == 0) {
      partials[base] = f0;
      partials[(base + 1)] = f1;
      partials[(base + 2)] = f2;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_fgmres_store_z_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_fgmres_store_z_f32(device const float* z [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* colIdx [[buffer(2)]], device float* Z [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int col = ((int)(colIdx[0]));
  if (gid < N) {
    float zi = z[gid];
    Z[(gid + (col * N))] = zi;
  }
}

)kernel";
const char* k_metal_navatala_sparse_fgmres_update_solution_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_fgmres_update_solution_f32(device const float* Z [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device float* x [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int M = ((int)(m[0]));
  if (gid < N) {
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(M); ++j) {
      float yj = y[j];
      float zij = Z[(gid + (j * N))];
      acc = (acc + (yj * zij));
    }
    float xi = x[gid];
    x[gid] = (xi + acc);
  }
}

)kernel";
const char* k_metal_navatala_sparse_pipe_c_g_fused_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_pipe_c_g_fused_f32(device const float* p [[buffer(0)]], device const float* Ap [[buffer(1)]], device const float* diagInv [[buffer(2)]], device const float* alpha [[buffer(3)]], device const uint* n [[buffer(4)]], device float* x [[buffer(5)]], device float* r [[buffer(6)]], device float* z [[buffer(7)]], device float* partials [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float a = alpha[0];
  bool inBounds = (gid < N);
  if (inBounds) {
    float pi = p[gid];
    float api = Ap[gid];
    float xi = x[gid];
    float ri = r[gid];
    float rnew = (ri - (a * api));
    x[gid] = (xi + (a * pi));
    r[gid] = rnew;
    float di = diagInv[gid];
    z[gid] = (di * rnew);
  }
  float sq = ((inBounds) ? ((r[gid] * r[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_chronopoulos_c_g_scalars_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_chronopoulos_c_g_scalars_f32(device const float* gamma [[buffer(0)]], device const float* delta [[buffer(1)]], device const float* gammaOld [[buffer(2)]], device float* alpha [[buffer(3)]], device float* beta [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float g = gamma[0];
  float d = delta[0];
  float go = gammaOld[0];
  float a = (g / d);
  float b = (g / go);
  alpha[0] = a;
  beta[0] = b;
}

)kernel";
const char* k_metal_navatala_sparse_fgmres_m_g_s_project_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_fgmres_m_g_s_project_f32(device const float* vi [[buffer(0)]], device const uint* n [[buffer(1)]], device float* w [[buffer(2)]], device float* hij [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  float prod = ((inBounds) ? ((w[gid] * vi[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(prod);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      hij[0] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_pbicgstab_fused_update_conv_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_pbicgstab_fused_update_conv_f32(device const float* phat [[buffer(0)]], device const float* shat [[buffer(1)]], device const float* s [[buffer(2)]], device const float* t [[buffer(3)]], device const float* alpha [[buffer(4)]], device const float* omega [[buffer(5)]], device const uint* n [[buffer(6)]], device float* x [[buffer(7)]], device float* r [[buffer(8)]], device float* rrPartials [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  if (inBounds) {
    float a = alpha[0];
    float om = omega[0];
    float ph = phat[gid];
    float sh = shat[gid];
    float si = s[gid];
    float ti = t[gid];
    float xi = x[gid];
    float ri = (si - (om * ti));
    x[gid] = (xi + ((a * ph) + (om * sh)));
    r[gid] = ri;
  }
  float sq = ((inBounds) ? ((r[gid] * r[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      rrPartials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_jacobi_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_jacobi_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const float* x [[buffer(5)]], device const float* omega [[buffer(6)]], device const uint* nRows [[buffer(7)]], device float* xNew [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float offDiag = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    float om = omega[0];
    float xr = x[row];
    float br = b[row];
    float dr = diag[row];
    xNew[row] = (((as_type<float>(0x3f800000u) - om) * xr) + (om * ((br - offDiag) / dr)));
  }
}

)kernel";
const char* k_metal_navatala_sparse_jacobi_l1_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_jacobi_l1_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* b [[buffer(3)]], device const float* x [[buffer(4)]], device const float* omega [[buffer(5)]], device const uint* nRows [[buffer(6)]], device float* xNew [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float offDiag = as_type<float>(0x00000000u);
    float l1Norm = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      l1Norm = (l1Norm + abs(a));
      if (col != row) {
        float xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    float om = omega[0];
    float xr = x[row];
    float br = b[row];
    xNew[row] = (xr + (om * ((br - offDiag) / l1Norm)));
  }
}

)kernel";
const char* k_metal_navatala_sparse_extract_diagonal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_extract_diagonal_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device float* diag [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col == row) {
        diag[row] = values[k];
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_block_jacobi_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_block_jacobi_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diagBlocks [[buffer(3)]], device const float* b [[buffer(4)]], device const float* x [[buffer(5)]], device const float* omega [[buffer(6)]], device const uint* nBlockRows [[buffer(7)]], device const uint* blockDim [[buffer(8)]], device float* xNew [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int blockRow = int(__gid.x);
  int N = ((int)(nBlockRows[0]));
  if (blockRow < N) {
    int bdm = ((int)(blockDim[0]));
    int bdm2 = (bdm * bdm);
    int rs = ((int)(rowPtr[blockRow]));
    int re = ((int)(rowPtr[(blockRow + 1)]));
    float om = omega[0];
    if (bdm == 4) {
      int rowBase4 = (blockRow * 4);
      float offDiag0 = as_type<float>(0x00000000u);
      float offDiag1 = as_type<float>(0x00000000u);
      float offDiag2 = as_type<float>(0x00000000u);
      float offDiag3 = as_type<float>(0x00000000u);
      for (int jj = 0; jj < (int)((re - rs)); ++jj) {
        int k = (rs + jj);
        int col = ((int)(colIdx[k]));
        if (col != blockRow) {
          int kBase = (k * 16);
          int colBase4 = (col * 4);
          float xv0 = x[colBase4];
          float xv1 = x[(colBase4 + 1)];
          float xv2 = x[(colBase4 + 2)];
          float xv3 = x[(colBase4 + 3)];
          offDiag0 = (offDiag0 + (((values[kBase] * xv0) + (values[(kBase + 1)] * xv1)) + ((values[(kBase + 2)] * xv2) + (values[(kBase + 3)] * xv3))));
          offDiag1 = (offDiag1 + (((values[(kBase + 4)] * xv0) + (values[(kBase + 5)] * xv1)) + ((values[(kBase + 6)] * xv2) + (values[(kBase + 7)] * xv3))));
          offDiag2 = (offDiag2 + (((values[(kBase + 8)] * xv0) + (values[(kBase + 9)] * xv1)) + ((values[(kBase + 10)] * xv2) + (values[(kBase + 11)] * xv3))));
          offDiag3 = (offDiag3 + (((values[(kBase + 12)] * xv0) + (values[(kBase + 13)] * xv1)) + ((values[(kBase + 14)] * xv2) + (values[(kBase + 15)] * xv3))));
        }
      }
      float bMinus0 = (b[rowBase4] - offDiag0);
      float bMinus1 = (b[(rowBase4 + 1)] - offDiag1);
      float bMinus2 = (b[(rowBase4 + 2)] - offDiag2);
      float bMinus3 = (b[(rowBase4 + 3)] - offDiag3);
      int dBase = (blockRow * 16);
      float xr0 = x[rowBase4];
      float xr1 = x[(rowBase4 + 1)];
      float xr2 = x[(rowBase4 + 2)];
      float xr3 = x[(rowBase4 + 3)];
      float oneMinusOm = (as_type<float>(0x3f800000u) - om);
      xNew[rowBase4] = ((oneMinusOm * xr0) + (om * (((diagBlocks[dBase] * bMinus0) + (diagBlocks[(dBase + 1)] * bMinus1)) + ((diagBlocks[(dBase + 2)] * bMinus2) + (diagBlocks[(dBase + 3)] * bMinus3)))));
      xNew[(rowBase4 + 1)] = ((oneMinusOm * xr1) + (om * (((diagBlocks[(dBase + 4)] * bMinus0) + (diagBlocks[(dBase + 5)] * bMinus1)) + ((diagBlocks[(dBase + 6)] * bMinus2) + (diagBlocks[(dBase + 7)] * bMinus3)))));
      xNew[(rowBase4 + 2)] = ((oneMinusOm * xr2) + (om * (((diagBlocks[(dBase + 8)] * bMinus0) + (diagBlocks[(dBase + 9)] * bMinus1)) + ((diagBlocks[(dBase + 10)] * bMinus2) + (diagBlocks[(dBase + 11)] * bMinus3)))));
      xNew[(rowBase4 + 3)] = ((oneMinusOm * xr3) + (om * (((diagBlocks[(dBase + 12)] * bMinus0) + (diagBlocks[(dBase + 13)] * bMinus1)) + ((diagBlocks[(dBase + 14)] * bMinus2) + (diagBlocks[(dBase + 15)] * bMinus3)))));
    } else {
      for (int eq = 0; eq < (int)(bdm); ++eq) {
        int grow = ((blockRow * bdm) + eq);
        float offDiag = as_type<float>(0x00000000u);
        for (int jj = 0; jj < (int)((re - rs)); ++jj) {
          int k = (rs + jj);
          int col = ((int)(colIdx[k]));
          if (col != blockRow) {
            int kBase = (k * bdm2);
            int rowBase = (kBase + (eq * bdm));
            for (int eq2 = 0; eq2 < (int)(bdm); ++eq2) {
              int grow2 = ((col * bdm) + eq2);
              float aval = values[(rowBase + eq2)];
              float xval = x[grow2];
              offDiag = (offDiag + (aval * xval));
            }
          }
        }
        float rhs = (b[grow] - offDiag);
        int invIdx = (((blockRow * bdm2) + (eq * bdm)) + eq);
        float invVal = diagBlocks[invIdx];
        float xr = x[grow];
        xNew[grow] = (((as_type<float>(0x3f800000u) - om) * xr) + (om * (invVal * rhs)));
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_jacobi_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_jacobi_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const float* x [[buffer(5)]], device const int* cfMarking [[buffer(6)]], device const float* omegaC [[buffer(7)]], device const float* omegaF [[buffer(8)]], device const uint* nRows [[buffer(9)]], device float* xNew [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float offDiag = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    int cf = cfMarking[row];
    float om = (((cf == 1)) ? (omegaC[0]) : (omegaF[0]));
    float xr = x[row];
    float br = b[row];
    float dr = diag[row];
    xNew[row] = (((as_type<float>(0x3f800000u) - om) * xr) + (om * ((br - offDiag) / dr)));
  }
}

)kernel";
const char* k_metal_navatala_sparse_multicolor_gs_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multicolor_gs_forward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* colorOffsets [[buffer(5)]], device const uint* nRowsThisColor [[buffer(6)]], device float* x [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int start = ((int)(colorOffsets[0]));
    int row = (start + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float bi = b[row];
    float di = diag[row];
    x[row] = ((bi - sum) / di);
  }
}

)kernel";
const char* k_metal_navatala_sparse_multicolor_gs_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multicolor_gs_backward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* colorOffsets [[buffer(5)]], device const uint* nRowsThisColor [[buffer(6)]], device float* x [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int start = ((int)(colorOffsets[0]));
    int row = (start + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float bi = b[row];
    float di = diag[row];
    x[row] = ((bi - sum) / di);
  }
}

)kernel";
const char* k_metal_navatala_sparse_fixcolor_gs_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_fixcolor_gs_forward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* reorderPerm [[buffer(5)]], device const uint* colorOffsets [[buffer(6)]], device const uint* nRowsThisColor [[buffer(7)]], device float* x [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int start = ((int)(colorOffsets[0]));
    int permIdx = (start + localIdx);
    int row = ((int)(reorderPerm[permIdx]));
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float bi = b[row];
    float di = diag[row];
    x[row] = ((bi - sum) / di);
  }
}

)kernel";
const char* k_metal_navatala_sparse_fixcolor_gs_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_fixcolor_gs_backward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* reorderPerm [[buffer(5)]], device const uint* colorOffsets [[buffer(6)]], device const uint* nRowsThisColor [[buffer(7)]], device float* x [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int start = ((int)(colorOffsets[0]));
    int permIdx = (start + localIdx);
    int row = ((int)(reorderPerm[permIdx]));
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float bi = b[row];
    float di = diag[row];
    x[row] = ((bi - sum) / di);
  }
}

)kernel";
const char* k_metal_navatala_sparse_ilu_factor_color_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ilu_factor_color_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* diagIdx [[buffer(2)]], device const uint* colorOffsets [[buffer(3)]], device const uint* nRowsThisColor [[buffer(4)]], device float* values [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    for (int jj = 0; jj < (int)((re - rs)); ++jj) {
      int k = (rs + jj);
      int col = ((int)(colIdx[k]));
      if (col < row) {
        int colDiagPos = ((int)(diagIdx[col]));
        float diagVal = values[colDiagPos];
        float lFactor = (values[k] / diagVal);
        values[k] = lFactor;
        int colRs = ((int)(rowPtr[col]));
        int colRe = ((int)(rowPtr[(col + 1)]));
        for (int mm = 0; mm < (int)((re - (k + 1))); ++mm) {
          int m = ((k + 1) + mm);
          int mcol = ((int)(colIdx[m]));
          for (int pp = 0; pp < (int)((colRe - colRs)); ++pp) {
            int p = (colRs + pp);
            int pcol = ((int)(colIdx[p]));
            if (pcol == mcol) {
              float uVal = values[p];
              float aim = values[m];
              values[m] = (aim - (lFactor * uVal));
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_ilu_forward_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ilu_forward_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* luValues [[buffer(2)]], device const float* b [[buffer(3)]], device const uint* colorOffsets [[buffer(4)]], device const uint* nRowsThisColor [[buffer(5)]], device float* y [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float bi = b[row];
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col < row) {
        float a = luValues[k];
        float yj = y[col];
        sum = (sum + (a * yj));
      }
    }
    y[row] = (bi - sum);
  }
}

)kernel";
const char* k_metal_navatala_sparse_ilu_backward_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ilu_backward_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* diagIdx [[buffer(2)]], device const float* luValues [[buffer(3)]], device const float* y [[buffer(4)]], device const uint* colorOffsets [[buffer(5)]], device const uint* nRowsThisColor [[buffer(6)]], device float* x [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float yi = y[row];
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col > row) {
        float a = luValues[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    int diagPos = ((int)(diagIdx[row]));
    float uDiag = luValues[diagPos];
    x[row] = ((yi - sum) / uDiag);
  }
}

)kernel";
const char* k_metal_navatala_sparse_dilu_build_reciprocal_d_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dilu_build_reciprocal_d_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* colorOffsets [[buffer(3)]], device const uint* nRowsThisColor [[buffer(4)]], device float* recipD [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float diag = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col == row) {
        diag = values[k];
      }
    }
    float sum = as_type<float>(0x00000000u);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      int col2 = ((int)(colIdx[k2]));
      if (col2 < row) {
        float aij = values[k2];
        float rdj = recipD[col2];
        int jrs = ((int)(rowPtr[col2]));
        int jre = ((int)(rowPtr[(col2 + 1)]));
        float aji = as_type<float>(0x00000000u);
        for (int pp = 0; pp < (int)((jre - jrs)); ++pp) {
          int p = (jrs + pp);
          int pcol = ((int)(colIdx[p]));
          if (pcol == row) {
            aji = values[p];
          }
        }
        sum = (sum + (aij * (aji * rdj)));
      }
    }
    recipD[row] = (as_type<float>(0x3f800000u) / (diag - sum));
  }
}

)kernel";
const char* k_metal_navatala_sparse_dilu_apply_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dilu_apply_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* recipD [[buffer(3)]], device const float* b [[buffer(4)]], device const float* x [[buffer(5)]], device const uint* colorOffsets [[buffer(6)]], device const uint* nRowsThisColor [[buffer(7)]], device float* xNew [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float bi = b[row];
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float rd = recipD[row];
    xNew[row] = (rd * (bi - sum));
  }
}

)kernel";
const char* k_metal_navatala_sparse_multicolor_dilu_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multicolor_dilu_forward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* recipD [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* colorOffsets [[buffer(5)]], device const uint* nRowsThisColor [[buffer(6)]], device float* x [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float bi = b[row];
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col < row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float rd = recipD[row];
    x[row] = (rd * (bi - sum));
  }
}

)kernel";
const char* k_metal_navatala_sparse_multicolor_dilu_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multicolor_dilu_backward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* recipD [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* colorOffsets [[buffer(5)]], device const uint* nRowsThisColor [[buffer(6)]], device float* x [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col > row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float rd = recipD[row];
    float bi = b[row];
    x[row] = (rd * (bi - sum));
  }
}

)kernel";
const char* k_metal_navatala_sparse_chebyshev_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_chebyshev_sweep_f32(device const float* r [[buffer(0)]], device const float* d [[buffer(1)]], device const float* coeffA [[buffer(2)]], device const float* coeffB [[buffer(3)]], device const uint* n [[buffer(4)]], device float* x [[buffer(5)]], device float* dNew [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if (i < N) {
    float cA = coeffA[0];
    float cB = coeffB[0];
    float ri = r[i];
    float di = d[i];
    float xi = x[i];
    float corr = ((cA * ri) + (cB * di));
    x[i] = (xi + corr);
    dNew[i] = corr;
  }
}

)kernel";
const char* k_metal_navatala_sparse_chebyshev_coeff_update_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_chebyshev_coeff_update_f32(device const float* theta [[buffer(0)]], device const float* delta [[buffer(1)]], device const float* sigma1 [[buffer(2)]], device const float* sigmaOld [[buffer(3)]], device float* coeffA [[buffer(4)]], device float* coeffB [[buffer(5)]], device float* sigmaNew [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float th = theta[0];
  float de = delta[0];
  float s1 = sigma1[0];
  float so = sigmaOld[0];
  float twoThOverDe = ((as_type<float>(0x40000000u) * th) / de);
  float sn = (as_type<float>(0x3f800000u) / (twoThOverDe - so));
  float cA = ((as_type<float>(0x40000000u) * sn) / (de * s1));
  float cB = (sn * (so - as_type<float>(0x3f800000u)));
  coeffA[0] = cA;
  coeffB[0] = cB;
  sigmaNew[0] = sn;
}

)kernel";
const char* k_metal_navatala_sparse_polynomial_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_polynomial_sweep_f32(device const float* Ax [[buffer(0)]], device const float* x [[buffer(1)]], device const float* b [[buffer(2)]], device const float* coeffs [[buffer(3)]], device const uint* degree [[buffer(4)]], device const uint* n [[buffer(5)]], device float* xNew [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if (i < N) {
    float ri = (b[i] - Ax[i]);
    float xi = x[i];
    int deg = ((int)(degree[0]));
    float poly = as_type<float>(0x00000000u);
    float power = as_type<float>(0x3f800000u);
    for (int k = 0; k < (int)(deg); ++k) {
      float ck = coeffs[k];
      poly = (poly + (ck * power));
      power = (power * ri);
    }
    xNew[i] = (xi + poly);
  }
}

)kernel";
const char* k_metal_navatala_sparse_kpz_polynomial_sweep_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_kpz_polynomial_sweep_f32(device const float* Ax [[buffer(0)]], device const float* x [[buffer(1)]], device const float* b [[buffer(2)]], device const float* kpzCoeffs [[buffer(3)]], device const uint* degree [[buffer(4)]], device const uint* n [[buffer(5)]], device float* xNew [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if (i < N) {
    float ri = (b[i] - Ax[i]);
    float xi = x[i];
    int deg = ((int)(degree[0]));
    float poly = as_type<float>(0x00000000u);
    float power = as_type<float>(0x3f800000u);
    for (int k = 0; k < (int)(deg); ++k) {
      float ck = kpzCoeffs[k];
      poly = (poly + (ck * power));
      power = (power * ri);
    }
    xNew[i] = (xi + poly);
  }
}

)kernel";
const char* k_metal_navatala_sparse_kaczmarz_row_norm_sq_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_kaczmarz_row_norm_sq_f32(device const uint* rowPtr [[buffer(0)]], device const float* values [[buffer(1)]], device const uint* nRows [[buffer(2)]], device float* rowNormSq [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float a = values[k];
      acc = (acc + (a * a));
    }
    rowNormSq[row] = acc;
  }
}

)kernel";
const char* k_metal_navatala_sparse_strength_of_connection_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_strength_of_connection_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* theta [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* strongMask [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float maxOff = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        float a = values[k];
        maxOff = (((abs(a) > maxOff)) ? (abs(a)) : (maxOff));
      }
    }
    float th = theta[0];
    float thresh = (th * maxOff);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float a2 = values[k2];
      uint isStrong = ((((((int)(colIdx[k2])) != gid) && (abs(a2) > thresh))) ? (1u) : (0u));
      strongMask[k2] = isStrong;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_size2_propose_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_size2_propose_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* pickArray [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int bestCol = -1;
    float bestVal = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        float a = values[k];
        if (abs(a) > bestVal) {
          bestVal = abs(a);
          bestCol = col;
        }
      }
    }
    pickArray[gid] = bestCol;
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_size2_confirm = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_aggregate_size2_confirm(device const int* pickArray [[buffer(0)]], device const uint* nRows [[buffer(1)]], device int* aggregateId [[buffer(2)]], device atomic_uint* nAggregates [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myPick = pickArray[gid];
    if (myPick < 0) {
      aggregateId[gid] = -1;
    } else {
      int peerPick = pickArray[myPick];
      if (peerPick == gid) {
        int leader = (((gid < myPick)) ? (gid) : (myPick));
        aggregateId[gid] = leader;
        if (gid < myPick) {
          uint _naBump = atomic_fetch_add_explicit(((device atomic_uint*)(&(nAggregates[0]))), 1u, memory_order_relaxed);
        }
      } else {
        aggregateId[gid] = -1;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_promote_unmatched_singletons = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_promote_unmatched_singletons(device const uint* nRows [[buffer(0)]], device int* aggregateId [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int aggId = aggregateId[gid];
    if (aggId < 0) {
      aggregateId[gid] = gid;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_size4_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_aggregate_size4_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* aggregateId [[buffer(5)]], device atomic_uint* nAggregates [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float bestVal_0 = as_type<float>(0x00000000u);
    int bestCol_0 = -1;
    float bestVal_1 = as_type<float>(0x00000000u);
    int bestCol_1 = -1;
    float bestVal_2 = as_type<float>(0x00000000u);
    int bestCol_2 = -1;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int kk = (rs + j);
      int col = ((int)(colIdx[kk]));
      uint isStrong = strongMask[kk];
      if (isStrong == 1u) {
        float a = values[kk];
        float aAbs = abs(a);
        float oldVal_0 = bestVal_0;
        int oldCol_0 = bestCol_0;
        float oldVal_1 = bestVal_1;
        int oldCol_1 = bestCol_1;
        float oldVal_2 = bestVal_2;
        int oldCol_2 = bestCol_2;
        bestVal_0 = (((aAbs > oldVal_0)) ? (aAbs) : (oldVal_0));
        bestCol_0 = (((aAbs > oldVal_0)) ? (col) : (oldCol_0));
        bestVal_1 = (((aAbs > oldVal_0)) ? (oldVal_0) : ((((aAbs > oldVal_1)) ? (aAbs) : (oldVal_1))));
        bestCol_1 = (((aAbs > oldVal_0)) ? (oldCol_0) : ((((aAbs > oldVal_1)) ? (col) : (oldCol_1))));
        bestVal_2 = (((aAbs > oldVal_1)) ? (oldVal_1) : ((((aAbs > oldVal_2)) ? (aAbs) : (oldVal_2))));
        bestCol_2 = (((aAbs > oldVal_1)) ? (oldCol_1) : ((((aAbs > oldVal_2)) ? (col) : (oldCol_2))));
      }
    }
    int aggId = (((((false || (bestCol_0 >= 0)) || (bestCol_1 >= 0)) || (bestCol_2 >= 0))) ? (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) : (-1));
    aggregateId[gid] = aggId;
    if (aggId == gid) {
      uint _naBump = atomic_fetch_add_explicit(((device atomic_uint*)(&(nAggregates[0]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_size8_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_aggregate_size8_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* aggregateId [[buffer(5)]], device atomic_uint* nAggregates [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float bestVal_0 = as_type<float>(0x00000000u);
    int bestCol_0 = -1;
    float bestVal_1 = as_type<float>(0x00000000u);
    int bestCol_1 = -1;
    float bestVal_2 = as_type<float>(0x00000000u);
    int bestCol_2 = -1;
    float bestVal_3 = as_type<float>(0x00000000u);
    int bestCol_3 = -1;
    float bestVal_4 = as_type<float>(0x00000000u);
    int bestCol_4 = -1;
    float bestVal_5 = as_type<float>(0x00000000u);
    int bestCol_5 = -1;
    float bestVal_6 = as_type<float>(0x00000000u);
    int bestCol_6 = -1;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int kk = (rs + j);
      int col = ((int)(colIdx[kk]));
      uint isStrong = strongMask[kk];
      if (isStrong == 1u) {
        float a = values[kk];
        float aAbs = abs(a);
        float oldVal_0 = bestVal_0;
        int oldCol_0 = bestCol_0;
        float oldVal_1 = bestVal_1;
        int oldCol_1 = bestCol_1;
        float oldVal_2 = bestVal_2;
        int oldCol_2 = bestCol_2;
        float oldVal_3 = bestVal_3;
        int oldCol_3 = bestCol_3;
        float oldVal_4 = bestVal_4;
        int oldCol_4 = bestCol_4;
        float oldVal_5 = bestVal_5;
        int oldCol_5 = bestCol_5;
        float oldVal_6 = bestVal_6;
        int oldCol_6 = bestCol_6;
        bestVal_0 = (((aAbs > oldVal_0)) ? (aAbs) : (oldVal_0));
        bestCol_0 = (((aAbs > oldVal_0)) ? (col) : (oldCol_0));
        bestVal_1 = (((aAbs > oldVal_0)) ? (oldVal_0) : ((((aAbs > oldVal_1)) ? (aAbs) : (oldVal_1))));
        bestCol_1 = (((aAbs > oldVal_0)) ? (oldCol_0) : ((((aAbs > oldVal_1)) ? (col) : (oldCol_1))));
        bestVal_2 = (((aAbs > oldVal_1)) ? (oldVal_1) : ((((aAbs > oldVal_2)) ? (aAbs) : (oldVal_2))));
        bestCol_2 = (((aAbs > oldVal_1)) ? (oldCol_1) : ((((aAbs > oldVal_2)) ? (col) : (oldCol_2))));
        bestVal_3 = (((aAbs > oldVal_2)) ? (oldVal_2) : ((((aAbs > oldVal_3)) ? (aAbs) : (oldVal_3))));
        bestCol_3 = (((aAbs > oldVal_2)) ? (oldCol_2) : ((((aAbs > oldVal_3)) ? (col) : (oldCol_3))));
        bestVal_4 = (((aAbs > oldVal_3)) ? (oldVal_3) : ((((aAbs > oldVal_4)) ? (aAbs) : (oldVal_4))));
        bestCol_4 = (((aAbs > oldVal_3)) ? (oldCol_3) : ((((aAbs > oldVal_4)) ? (col) : (oldCol_4))));
        bestVal_5 = (((aAbs > oldVal_4)) ? (oldVal_4) : ((((aAbs > oldVal_5)) ? (aAbs) : (oldVal_5))));
        bestCol_5 = (((aAbs > oldVal_4)) ? (oldCol_4) : ((((aAbs > oldVal_5)) ? (col) : (oldCol_5))));
        bestVal_6 = (((aAbs > oldVal_5)) ? (oldVal_5) : ((((aAbs > oldVal_6)) ? (aAbs) : (oldVal_6))));
        bestCol_6 = (((aAbs > oldVal_5)) ? (oldCol_5) : ((((aAbs > oldVal_6)) ? (col) : (oldCol_6))));
      }
    }
    int aggId = (((((((((false || (bestCol_0 >= 0)) || (bestCol_1 >= 0)) || (bestCol_2 >= 0)) || (bestCol_3 >= 0)) || (bestCol_4 >= 0)) || (bestCol_5 >= 0)) || (bestCol_6 >= 0))) ? (((((bestCol_6 >= 0) && (bestCol_6 < ((((bestCol_5 >= 0) && (bestCol_5 < ((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))) ? (bestCol_5) : (((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))))) ? (bestCol_6) : (((((bestCol_5 >= 0) && (bestCol_5 < ((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))) ? (bestCol_5) : (((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))))) : (-1));
    aggregateId[gid] = aggId;
    if (aggId == gid) {
      uint _naBump = atomic_fetch_add_explicit(((device atomic_uint*)(&(nAggregates[0]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_parallel_greedy_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_parallel_greedy_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* seed [[buffer(5)]], device int* aggregateId [[buffer(6)]], device uint* nAggregates [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int bestCol = -1;
    float bestVal = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        float a = values[k];
        if (abs(a) > bestVal) {
          bestVal = abs(a);
          bestCol = ((int)(colIdx[k]));
        }
      }
    }
    if (bestCol >= 0) {
      int aggVal = (((gid < bestCol)) ? (gid) : (bestCol));
      aggregateId[gid] = aggVal;
      aggregateId[bestCol] = aggVal;
    } else {
      aggregateId[gid] = gid;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_assign_leader_compact_ids = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_assign_leader_compact_ids(device const uint* nRows [[buffer(0)]], device const int* aggregateId [[buffer(1)]], device int* compactMap [[buffer(2)]], device atomic_uint* nAggregates [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int aggId = aggregateId[gid];
    if (aggId == gid) {
      uint compactId = atomic_fetch_add_explicit(((device atomic_uint*)(&(nAggregates[0]))), 1u, memory_order_relaxed);
      compactMap[gid] = ((int)(compactId));
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_apply_compact_ids = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_apply_compact_ids(device const int* compactMap [[buffer(0)]], device const uint* nRows [[buffer(1)]], device int* aggregateId [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int leader = aggregateId[gid];
    if (leader >= 0) {
      int compactId = compactMap[leader];
      aggregateId[gid] = compactId;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_prolongation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_build_prolongation_f32(device const int* aggregateId [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* ProwPtr [[buffer(2)]], device uint* PcolIdx [[buffer(3)]], device float* Pvalues [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    ProwPtr[row] = ((uint)(row));
    int aggId = aggregateId[row];
    PcolIdx[row] = ((uint)(aggId));
    Pvalues[row] = as_type<float>(0x3f800000u);
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_prolongation_terminator = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_build_prolongation_terminator(device const uint* nRows [[buffer(0)]], device uint* ProwPtr [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    uint N = nRows[0];
    ProwPtr[N] = N;
  }
}

)kernel";
const char* k_metal_navatala_sparse_spgemm_symbolic_row_nnz = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_spgemm_symbolic_row_nnz(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const uint* BrowPtr [[buffer(2)]], device const uint* BcolIdx [[buffer(3)]], device const uint* nRowsA [[buffer(4)]], device uint* CrowNnz [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRowsA[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int count = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      count = (count + (bre - brs));
    }
    CrowNnz[gid] = ((uint)(count));
  }
}

)kernel";
const char* k_metal_navatala_sparse_spgemm_numeric_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_spgemm_numeric_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const uint* BrowPtr [[buffer(3)]], device const uint* BcolIdx [[buffer(4)]], device const float* Bvalues [[buffer(5)]], device const uint* CrowPtr [[buffer(6)]], device const uint* nRowsA [[buffer(7)]], device uint* CcolIdx [[buffer(8)]], device float* Cvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRowsA[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int crs = ((int)(CrowPtr[gid]));
    int ci = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      float aVal = Avalues[ak];
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      for (int jj = 0; jj < (int)((bre - brs)); ++jj) {
        int bk = (brs + jj);
        int tgt = (crs + ci);
        uint bColIdx = BcolIdx[bk];
        float bVal = Bvalues[bk];
        CcolIdx[tgt] = bColIdx;
        Cvalues[tgt] = (aVal * bVal);
        ci = (ci + 1);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_transpose = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_transpose(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* nCols [[buffer(4)]], device const uint* TrowPtr [[buffer(5)]], device uint* TcolIdx [[buffer(6)]], device float* Tvalues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float val = values[k];
      int tp = ((int)(TrowPtr[col]));
      int tgt = (tp + j);
      TcolIdx[tgt] = ((uint)(gid));
      Tvalues[tgt] = val;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_adaptive_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_adaptive_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* degreeThreshold [[buffer(5)]], device int* aggregateId [[buffer(6)]], device uint* nAggregates [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int degree = (re - rs);
    int thresh = ((int)(degreeThreshold[0]));
    int bestCol = -1;
    float bestVal = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        float a = values[k];
        if (abs(a) > bestVal) {
          bestVal = abs(a);
          bestCol = ((int)(colIdx[k]));
        }
      }
    }
    if (bestCol >= 0) {
      aggregateId[gid] = (((gid < bestCol)) ? (gid) : (bestCol));
    } else {
      aggregateId[gid] = gid;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_multi_pairwise_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_multi_pairwise_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* maxPasses [[buffer(5)]], device int* aggregateId [[buffer(6)]], device uint* nAggregates [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int curAgg = aggregateId[gid];
    if (curAgg < 0) {
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int bestCol = -1;
      float bestVal = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStrong = strongMask[k];
        if (isStrong == 1u) {
          float a = values[k];
          if (abs(a) > bestVal) {
            bestVal = abs(a);
            bestCol = ((int)(colIdx[k]));
          }
        }
      }
      if (bestCol >= 0) {
        aggregateId[gid] = (((gid < bestCol)) ? (gid) : (bestCol));
      } else {
        aggregateId[gid] = gid;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_geometric_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_geometric_f32(device const float* coordX [[buffer(0)]], device const float* coordY [[buffer(1)]], device const float* coordZ [[buffer(2)]], device const float* radius [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* aggregateId [[buffer(5)]], device uint* nAggregates [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    float r = radius[0];
    aggregateId[gid] = gid;
  }
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_serial_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_serial_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* aggregateId [[buffer(5)]], device uint* nAggregates [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int N = ((int)(nRows[0]));
  int nextAgg = 0;
  for (int i = 0; i < (int)(N); ++i) {
    int rs = ((int)(rowPtr[i]));
    int re = ((int)(rowPtr[(i + 1)]));
    int bestCol = -1;
    float bestVal = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        float a = values[k];
        if (abs(a) > bestVal) {
          bestVal = abs(a);
          bestCol = ((int)(colIdx[k]));
        }
      }
    }
    if (bestCol >= 0) {
      aggregateId[i] = (((i < bestCol)) ? (i) : (bestCol));
    } else {
      aggregateId[i] = nextAgg;
      nextAgg = (nextAgg + 1);
    }
  }
  nAggregates[0] = ((uint)(nextAgg));
}

)kernel";
const char* k_metal_navatala_sparse_aggregate_low_degree_first_propose_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_aggregate_low_degree_first_propose_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* pickArray [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int bestCol = -1;
    float bestVal = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        float a = values[k];
        if (abs(a) > bestVal) {
          bestVal = abs(a);
          bestCol = col;
        }
      }
    }
    pickArray[gid] = bestCol;
  }
}

)kernel";
const char* k_metal_navatala_sparse_assign_orphans_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_assign_orphans_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device int* aggregateId [[buffer(4)]], device uint* nAggregates [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int curAgg = aggregateId[gid];
    if (curAgg < 0) {
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int bestAgg = -1;
      float bestVal = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        int neighAgg = aggregateId[col];
        if (neighAgg >= 0) {
          float a = values[k];
          if (abs(a) > bestVal) {
            bestVal = abs(a);
            bestAgg = neighAgg;
          }
        }
      }
      if (bestAgg >= 0) {
        aggregateId[gid] = bestAgg;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_spgemm_numeric_hash_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_spgemm_numeric_hash_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const uint* BrowPtr [[buffer(3)]], device const uint* BcolIdx [[buffer(4)]], device const float* Bvalues [[buffer(5)]], device const uint* CrowPtr [[buffer(6)]], device const uint* nRowsA [[buffer(7)]], device const uint* hashSize [[buffer(8)]], device uint* CcolIdx [[buffer(9)]], device float* Cvalues [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRowsA[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int crs = ((int)(CrowPtr[gid]));
    int ci = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      float aVal = Avalues[ak];
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      for (int jj = 0; jj < (int)((bre - brs)); ++jj) {
        int bk = (brs + jj);
        int tgt = (crs + ci);
        uint bColIdx = BcolIdx[bk];
        float bVal = Bvalues[bk];
        CcolIdx[tgt] = bColIdx;
        Cvalues[tgt] = (aVal * bVal);
        ci = (ci + 1);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_spgemm_numeric_serial_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_spgemm_numeric_serial_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const uint* BrowPtr [[buffer(3)]], device const uint* BcolIdx [[buffer(4)]], device const float* Bvalues [[buffer(5)]], device const uint* CrowPtr [[buffer(6)]], device const uint* nRowsA [[buffer(7)]], device uint* CcolIdx [[buffer(8)]], device float* Cvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int N = ((int)(nRowsA[0]));
  for (int row = 0; row < (int)(N); ++row) {
    int ars = ((int)(ArowPtr[row]));
    int are = ((int)(ArowPtr[(row + 1)]));
    int crs = ((int)(CrowPtr[row]));
    int ci = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      float aVal = Avalues[ak];
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      for (int jj = 0; jj < (int)((bre - brs)); ++jj) {
        int bk = (brs + jj);
        int tgt = (crs + ci);
        CcolIdx[tgt] = BcolIdx[bk];
        Cvalues[tgt] = (aVal * Bvalues[bk]);
        ci = (ci + 1);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_smoothed_prolongation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_build_smoothed_prolongation_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const float* diagInv [[buffer(3)]], device const uint* ProwPtr [[buffer(4)]], device const uint* PcolIdx [[buffer(5)]], device const float* Pvalues [[buffer(6)]], device const float* omega [[buffer(7)]], device const uint* nRows [[buffer(8)]], device float* PsmoothValues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    uint myAgg = PcolIdx[row];
    float myPval = Pvalues[row];
    int aStart = ((int)(ArowPtr[row]));
    int aEnd = ((int)(ArowPtr[(row + 1)]));
    float sumAP = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)((aEnd - aStart)); ++k) {
      int idx = (aStart + k);
      int col = ((int)(AcolIdx[idx]));
      float aVal = Avalues[idx];
      uint colAgg = PcolIdx[col];
      if (colAgg == myAgg) {
        sumAP = (sumAP + aVal);
      }
    }
    float om = omega[0];
    float di = diagInv[row];
    float correction = (om * (di * sumAP));
    PsmoothValues[row] = (myPval - correction);
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_aggregate_sizes = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_compute_aggregate_sizes(device const int* aggregateId [[buffer(0)]], device const uint* nRows [[buffer(1)]], device const uint* nAggregates [[buffer(2)]], device atomic_uint* aggSizes [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int aggId = aggregateId[gid];
    if (aggId >= 0) {
      uint _aod1 = atomic_fetch_add_explicit(((device atomic_uint*)(&(aggSizes[aggId]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_coarsening_ratio = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_compute_coarsening_ratio(device const uint* nFine [[buffer(0)]], device const uint* nCoarse [[buffer(1)]], device float* ratio [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float fine = ((float)(nFine[0]));
  float coarse = ((float)(nCoarse[0]));
  ratio[0] = (coarse / fine);
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_p_m_i_s = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_p_m_i_s(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* seed [[buffer(4)]], device int* cfMarking [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    uint priority = (((uint)(row)) ^ seed[0]);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    bool hasHigherNbr = false;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        int nbr = ((int)(colIdx[k]));
        int nbrMark = cfMarking[nbr];
        if (nbrMark == 0) {
          uint nbrPri = (((uint)(nbr)) ^ seed[0]);
          bool outranks = ((nbrPri > priority) || ((nbrPri == priority) && (nbr < row)));
          if (outranks) {
            hasHigherNbr = true;
          }
        }
      }
    }
    if (!hasHigherNbr) {
      cfMarking[row] = 1;
    } else {
      bool hasCoarseNbr = false;
      for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
        int k2 = (rs + j2);
        uint isStrong2 = strongMask[k2];
        if (isStrong2 == 1u) {
          int nbr2 = ((int)(colIdx[k2]));
          int nbrMark2 = cfMarking[nbr2];
          if (nbrMark2 == 1) {
            hasCoarseNbr = true;
          }
        }
      }
      if (hasCoarseNbr) {
        cfMarking[row] = -1;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_h_m_i_s = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_h_m_i_s(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* seed [[buffer(4)]], device int* cfMarking [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    uint priority = (((uint)(row)) ^ seed[0]);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    uint deg = 0u;
    for (int jd = 0; jd < (int)((re - rs)); ++jd) {
      int kd = (rs + jd);
      uint isStr = strongMask[kd];
      if (isStr == 1u) {
        deg = (deg + 1u);
      }
    }
    uint boostedPri = (priority + deg);
    bool hasHigherNbr = false;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        int nbr = ((int)(colIdx[k]));
        int nbrMark = cfMarking[nbr];
        if (nbrMark == 0) {
          uint nbrRs = rowPtr[nbr];
          uint nbrRe = rowPtr[(nbr + 1)];
          uint nbrDeg = (nbrRe - nbrRs);
          uint nbrPri = ((((uint)(nbr)) ^ seed[0]) + nbrDeg);
          bool outranks = ((nbrPri > boostedPri) || ((nbrPri == boostedPri) && (nbr < row)));
          if (outranks) {
            hasHigherNbr = true;
          }
        }
      }
    }
    if (!hasHigherNbr) {
      cfMarking[row] = 1;
    } else {
      cfMarking[row] = -1;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_h_m_i_s_phase2_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_h_m_i_s_phase2_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const uint* nRows [[buffer(3)]], device int* cfMarking [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == -1) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      bool hasCoarseNbr = false;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStrong = strongMask[k];
        if (isStrong == 1u) {
          int nbr = ((int)(colIdx[k]));
          int nbrMark = cfMarking[nbr];
          if (nbrMark == 1) {
            hasCoarseNbr = true;
          }
        }
      }
      if (hasCoarseNbr) {
      } else {
        cfMarking[row] = 1;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_r_s = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_r_s(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const uint* nRows [[buffer(3)]], device int* cfMarking [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int N = ((int)(nRows[0]));
  if (int(__gid.x) == 0) {
    for (int row = 0; row < (int)(N); ++row) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      uint nStrongC = 0u;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStrong = strongMask[k];
        if (isStrong == 1u) {
          int nbr = ((int)(colIdx[k]));
          int nbrMark = cfMarking[nbr];
          if (nbrMark == 1) {
            nStrongC = (nStrongC + 1u);
          }
        }
      }
      if (nStrongC == 0u) {
        cfMarking[row] = 1;
      } else {
        cfMarking[row] = -1;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_distance1_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_distance1_count_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int jc = 0; jc < (int)((re - rs)); ++jc) {
          int kc = (rs + jc);
          uint isStr = strongMask[kc];
          if (isStr == 1u) {
            int colC = ((int)(colIdx[kc]));
            int markC = cfMarking[colC];
            if (markC == 1) {
              nnz = (nnz + 1u);
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_distance1_count_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_distance1_count_f64(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int jc = 0; jc < (int)((re - rs)); ++jc) {
          int kc = (rs + jc);
          uint isStr = strongMask[kc];
          if (isStr == 1u) {
            int colC = ((int)(colIdx[kc]));
            int markC = cfMarking[colC];
            if (markC == 1) {
              nnz = (nnz + 1u);
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_distance1_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_distance1_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const uint* ProwPtr [[buffer(5)]], device const uint* nRows [[buffer(6)]], device uint* PcolIdx [[buffer(7)]], device float* Pvalues [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int mark = cfMarking[row];
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_type<float>(0x3f800000u);
    } else {
      if (mark == -1) {
        float diag = as_type<float>(0x00000000u);
        for (int jd = 0; jd < (int)((re - rs)); ++jd) {
          int kd = (rs + jd);
          int colD = ((int)(colIdx[kd]));
          if (colD == row) {
            diag = values[kd];
          }
        }
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int markW = cfMarking[colW];
            if (markW == 1) {
              float aij = values[kw];
              float w = (-(aij / diag));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_distance2_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_distance2_count_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if (isStr1 == 1u) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if (m1 == 1) {
              nnz = (nnz + 1u);
            } else {
              if (m1 == -1) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if (isStr2 == 1u) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if (m2 == 1) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_distance2_count_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_distance2_count_f64(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if (isStr1 == 1u) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if (m1 == 1) {
              nnz = (nnz + 1u);
            } else {
              if (m1 == -1) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if (isStr2 == 1u) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if (m2 == 1) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_distance2_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_distance2_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const uint* diagIdx [[buffer(5)]], device const uint* ProwPtr [[buffer(6)]], device const uint* nRows [[buffer(7)]], device uint* PcolIdx [[buffer(8)]], device float* Pvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_type<float>(0x3f800000u);
    } else {
      if (mark == -1) {
        int iDiagPos = ((int)(diagIdx[row]));
        float diagI = values[iDiagPos];
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int mW = cfMarking[colW];
            if (mW == 1) {
              float aij = values[kw];
              float w = (-(aij / diagI));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
        for (int jf = 0; jf < (int)((re - rs)); ++jf) {
          int kf = (rs + jf);
          uint isStrF = strongMask[kf];
          if (isStrF == 1u) {
            int nbrF = ((int)(colIdx[kf]));
            int mF = cfMarking[nbrF];
            if (mF == -1) {
              float aij = values[kf];
              int jDiagPos = ((int)(diagIdx[nbrF]));
              float diagJ = values[jDiagPos];
              int rs2 = ((int)(rowPtr[nbrF]));
              int re2 = ((int)(rowPtr[(nbrF + 1)]));
              for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                int k2 = (rs2 + j2);
                uint isStr2 = strongMask[k2];
                if (isStr2 == 1u) {
                  int nbr2 = ((int)(colIdx[k2]));
                  int m2 = cfMarking[nbr2];
                  if (m2 == 1) {
                    float ajk = values[k2];
                    float prod = (aij * (ajk / diagJ));
                    float wff = (-(prod / diagI));
                    PcolIdx[(pStart + pos)] = ((uint)(nbr2));
                    Pvalues[(pStart + pos)] = wff;
                    pos = (pos + 1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_truncate_interpolation_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_truncate_interpolation_count_f32(device const uint* ProwPtr [[buffer(0)]], device const uint* PcolIdx [[buffer(1)]], device const float* Pvalues [[buffer(2)]], device const float* threshold [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRowOut [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float maxAbs = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float v = Pvalues[k];
      maxAbs = (((abs(v) > maxAbs)) ? (abs(v)) : (maxAbs));
    }
    float thresh = (maxAbs * threshold[0]);
    uint nnz = 0u;
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float v2 = Pvalues[k2];
      if (abs(v2) >= thresh) {
        nnz = (nnz + 1u);
      }
    }
    nnzPerRowOut[row] = nnz;
  }
}

)kernel";
const char* k_metal_navatala_sparse_truncate_interpolation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_truncate_interpolation_f32(device const uint* ProwPtr [[buffer(0)]], device const uint* PcolIdx [[buffer(1)]], device const float* Pvalues [[buffer(2)]], device const float* threshold [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* ProwPtrOut [[buffer(5)]], device uint* PcolIdxOut [[buffer(6)]], device float* PvaluesOut [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    int pStart = ((int)(ProwPtrOut[row]));
    float maxAbs = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float v = Pvalues[k];
      maxAbs = (((abs(v) > maxAbs)) ? (abs(v)) : (maxAbs));
    }
    float thresh = (maxAbs * threshold[0]);
    int pos = 0;
    for (int j3 = 0; j3 < (int)((re - rs)); ++j3) {
      int k3 = (rs + j3);
      float v3 = Pvalues[k3];
      if (abs(v3) >= thresh) {
        PcolIdxOut[(pStart + pos)] = PcolIdx[k3];
        PvaluesOut[(pStart + pos)] = v3;
        pos = (pos + 1);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_aggressive_p_m_i_s = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_aggressive_p_m_i_s(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarkingPrev [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* seed [[buffer(5)]], device int* cfMarking [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int prevMark = cfMarkingPrev[row];
    if (prevMark == 1) {
      uint priority = (((uint)(row)) ^ seed[0]);
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      bool hasHigherNbr = false;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == 1u) {
          int nbr = ((int)(colIdx[k]));
          int nbrPrev = cfMarkingPrev[nbr];
          if (nbrPrev == 1) {
            uint nbrPri = (((uint)(nbr)) ^ seed[0]);
            bool outranks = ((nbrPri > priority) || ((nbrPri == priority) && (nbr < row)));
            if (outranks) {
              hasHigherNbr = true;
            }
          }
        }
      }
      if (!hasHigherNbr) {
        cfMarking[row] = 1;
      } else {
        cfMarking[row] = -1;
      }
    } else {
      cfMarking[row] = prevMark;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_aggressive_h_m_i_s = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_aggressive_h_m_i_s(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarkingPrev [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* seed [[buffer(5)]], device int* cfMarking [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int prevMark = cfMarkingPrev[row];
    if (prevMark == 1) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      uint deg = 0u;
      for (int jd = 0; jd < (int)((re - rs)); ++jd) {
        int kd = (rs + jd);
        uint isStr = strongMask[kd];
        if (isStr == 1u) {
          int nbrD = ((int)(colIdx[kd]));
          int nbrPD = cfMarkingPrev[nbrD];
          if (nbrPD == 1) {
            deg = (deg + 1u);
          }
        }
      }
      uint priority = ((((uint)(row)) ^ seed[0]) + deg);
      bool hasHigherNbr = false;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == 1u) {
          int nbr = ((int)(colIdx[k]));
          int nbrP = cfMarkingPrev[nbr];
          if (nbrP == 1) {
            uint nbrRs = rowPtr[nbr];
            uint nbrRe = rowPtr[(nbr + 1)];
            uint nbrDeg = (nbrRe - nbrRs);
            uint nbrPri = ((((uint)(nbr)) ^ seed[0]) + nbrDeg);
            bool outranks = ((nbrPri > priority) || ((nbrPri == priority) && (nbr < row)));
            if (outranks) {
              hasHigherNbr = true;
            }
          }
        }
      }
      if (!hasHigherNbr) {
        cfMarking[row] = 1;
      } else {
        cfMarking[row] = -1;
      }
    } else {
      cfMarking[row] = prevMark;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_c_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_c_r_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const float* crThreshold [[buffer(5)]], device int* cfMarking [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 0) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      float diag = as_type<float>(0x00000000u);
      float offDiagSum = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        float a = values[k];
        if (col == row) {
          diag = a;
        } else {
          offDiagSum = (offDiagSum + a);
        }
      }
      float crVal = abs((offDiagSum / diag));
      float crThresh = crThreshold[0];
      if (crVal > crThresh) {
        cfMarking[row] = 1;
      } else {
        bool hasC = false;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          int k2 = (rs + j2);
          uint isStr = strongMask[k2];
          if (isStr == 1u) {
            int nbr = ((int)(colIdx[k2]));
            int nbrM = cfMarking[nbr];
            if (nbrM == 1) {
              hasC = true;
            }
          }
        }
        if (hasC) {
          cfMarking[row] = -1;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_cf_split_c_l_j_p = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_cf_split_c_l_j_p(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const float* weights [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* seed [[buffer(5)]], device int* cfMarking [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    uint hash = (((uint)(row)) ^ seed[0]);
    float w = weights[row];
    uint priority = (hash + ((uint)(w)));
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    bool isHighest = true;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStr = strongMask[k];
      if (isStr == 1u) {
        int nbr = ((int)(colIdx[k]));
        uint nbrHash = (((uint)(nbr)) ^ seed[0]);
        float nbrW = weights[nbr];
        uint nbrPri = (nbrHash + ((uint)(nbrW)));
        if (nbrPri > priority) {
          isHighest = false;
        }
      }
    }
    if (isHighest) {
      cfMarking[row] = 1;
    } else {
      bool hasCoarse = false;
      for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
        int k2 = (rs + j2);
        uint isStr2 = strongMask[k2];
        if (isStr2 == 1u) {
          int nbr2 = ((int)(colIdx[k2]));
          int nbrM = cfMarking[nbr2];
          if (nbrM == 1) {
            hasCoarse = true;
          }
        }
      }
      if (hasCoarse) {
        cfMarking[row] = -1;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_multipass_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_multipass_count_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if (isStr1 == 1u) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if (m1 == 1) {
              nnz = (nnz + 1u);
            } else {
              if (m1 == -1) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if (isStr2 == 1u) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if (m2 == 1) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_multipass_count_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_multipass_count_f64(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if (isStr1 == 1u) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if (m1 == 1) {
              nnz = (nnz + 1u);
            } else {
              if (m1 == -1) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if (isStr2 == 1u) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if (m2 == 1) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_multipass_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_multipass_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const uint* diagIdx [[buffer(5)]], device const uint* ProwPtr [[buffer(6)]], device const uint* nRows [[buffer(7)]], device const uint* maxPass [[buffer(8)]], device uint* PcolIdx [[buffer(9)]], device float* Pvalues [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_type<float>(0x3f800000u);
    } else {
      if (mark == -1) {
        int iDiagPos = ((int)(diagIdx[row]));
        float diagI = values[iDiagPos];
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int mW = cfMarking[colW];
            if (mW == 1) {
              float aij = values[kw];
              float w = (-(aij / diagI));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
        for (int jf = 0; jf < (int)((re - rs)); ++jf) {
          int kf = (rs + jf);
          uint isStrF = strongMask[kf];
          if (isStrF == 1u) {
            int nbrF = ((int)(colIdx[kf]));
            int mF = cfMarking[nbrF];
            if (mF == -1) {
              float aij = values[kf];
              int jDiagPos = ((int)(diagIdx[nbrF]));
              float diagJ = values[jDiagPos];
              int rs2 = ((int)(rowPtr[nbrF]));
              int re2 = ((int)(rowPtr[(nbrF + 1)]));
              for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                int k2 = (rs2 + j2);
                uint isStr2 = strongMask[k2];
                if (isStr2 == 1u) {
                  int nbr2 = ((int)(colIdx[k2]));
                  int m2 = cfMarking[nbr2];
                  if (m2 == 1) {
                    float ajk = values[k2];
                    float prod = (aij * (ajk / diagJ));
                    float wff = (-(prod / diagI));
                    PcolIdx[(pStart + pos)] = ((uint)(nbr2));
                    Pvalues[(pStart + pos)] = wff;
                    pos = (pos + 1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_strength_symmetric_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_strength_symmetric_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* diagIdx [[buffer(3)]], device const float* theta [[buffer(4)]], device const uint* nRows [[buffer(5)]], device uint* strongMask [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int iDiagPos = ((int)(diagIdx[gid]));
    float diag = values[iDiagPos];
    float absDiagI = abs(diag);
    float thetaVal = theta[0];
    float theta2 = (thetaVal * thetaVal);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float a2 = (a * a);
      int jDiagPos = ((int)(diagIdx[col]));
      float absDiagJ = abs(values[jDiagPos]);
      float threshold = (theta2 * (absDiagI * absDiagJ));
      uint isStrong = ((((col != gid) && (a2 >= threshold))) ? (1u) : (0u));
      strongMask[k] = isStrong;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_strength_algebraic_dist_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_strength_algebraic_dist_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* testVectors [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* nTestVectors [[buffer(5)]], device const float* theta [[buffer(6)]], device uint* strongMask [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nTv = ((int)(nTestVectors[0]));
    float thetaVal = theta[0];
    float maxDist = as_type<float>(0x00000000u);
    for (int jm = 0; jm < (int)((re - rs)); ++jm) {
      int km = (rs + jm);
      int colM = ((int)(colIdx[km]));
      if (colM != gid) {
        float distM = as_type<float>(0x00000000u);
        for (int tvM = 0; tvM < (int)(nTv); ++tvM) {
          float viM = testVectors[((gid * nTv) + tvM)];
          float vjM = testVectors[((colM * nTv) + tvM)];
          distM = (distM + abs((viM - vjM)));
        }
        float avgDistM = (distM / ((float)(nTv)));
        maxDist = (((avgDistM > maxDist)) ? (avgDistM) : (maxDist));
      }
    }
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        float dist = as_type<float>(0x00000000u);
        for (int tv = 0; tv < (int)(nTv); ++tv) {
          float vi = testVectors[((gid * nTv) + tv)];
          float vj = testVectors[((col * nTv) + tv)];
          dist = (dist + abs((vi - vj)));
        }
        float avgDist = (dist / ((float)(nTv)));
        uint hasMax = (((maxDist > as_type<float>(0x00000000u))) ? (1u) : (0u));
        float normDist = (((hasMax == 1u)) ? ((dist / maxDist)) : (as_type<float>(0x00000000u)));
        float _unusedAvg = avgDist;
        uint isStrong = ((((hasMax == 1u) && (normDist <= thetaVal))) ? (1u) : (0u));
        strongMask[k] = isStrong;
      } else {
        strongMask[k] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_strength_affinity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_strength_affinity_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* theta [[buffer(3)]], device const uint* nRows [[buffer(4)]], device float* strengthValues [[buffer(5)]], device uint* strongMask [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float diag = as_type<float>(0x00000000u);
    for (int jd = 0; jd < (int)((re - rs)); ++jd) {
      int kd = (rs + jd);
      int colD = ((int)(colIdx[kd]));
      if (colD == gid) {
        diag = values[kd];
      }
    }
    float thetaVal = theta[0];
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float aff = (((col != gid)) ? ((abs(a) / abs(diag))) : (as_type<float>(0x00000000u)));
      strengthValues[k] = aff;
      uint isStrong = (((aff >= thetaVal)) ? (1u) : (0u));
      strongMask[k] = isStrong;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_count_c_f_points = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_count_c_f_points(device const int* cfMarking [[buffer(0)]], device const uint* nRows [[buffer(1)]], device atomic_uint* nCoarse [[buffer(2)]], device atomic_uint* nFine [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int mark = cfMarking[gid];
    if (mark == 1) {
      uint _nc = atomic_fetch_add_explicit(((device atomic_uint*)(&(nCoarse[0]))), 1u, memory_order_relaxed);
    }
    if (mark == -1) {
      uint _nf = atomic_fetch_add_explicit(((device atomic_uint*)(&(nFine[0]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_grid_complexity = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_compute_grid_complexity(device const uint* levelNnz [[buffer(0)]], device const uint* nLevels [[buffer(1)]], device float* gridComplexity [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    int nLev = ((int)(nLevels[0]));
    uint finestNnz = levelNnz[0];
    float total = as_type<float>(0x00000000u);
    for (int lev = 0; lev < (int)(nLev); ++lev) {
      uint nnzL = levelNnz[lev];
      total = (total + ((float)(nnzL)));
    }
    float gc = (total / ((float)(finestNnz)));
    gridComplexity[0] = gc;
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_operator_complexity = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_compute_operator_complexity(device const uint* levelNnz [[buffer(0)]], device const uint* levelRows [[buffer(1)]], device const uint* nLevels [[buffer(2)]], device float* operatorComplexity [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    int nLev = ((int)(nLevels[0]));
    uint finestNnz = levelNnz[0];
    float total = as_type<float>(0x00000000u);
    for (int lev = 0; lev < (int)(nLev); ++lev) {
      uint nnzL = levelNnz[lev];
      total = (total + ((float)(nnzL)));
    }
    float oc = (total / ((float)(finestNnz)));
    operatorComplexity[0] = oc;
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_ext_no_i_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_ext_no_i_count_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if (isStr1 == 1u) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if (m1 == 1) {
              nnz = (nnz + 1u);
            } else {
              if (m1 == -1) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if (isStr2 == 1u) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if (m2 == 1) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_ext_no_i_count_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_ext_no_i_count_f64(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      nnzPerRow[row] = 1u;
    } else {
      if (mark == -1) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if (isStr1 == 1u) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if (m1 == 1) {
              nnz = (nnz + 1u);
            } else {
              if (m1 == -1) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if (isStr2 == 1u) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if (m2 == 1) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_ext_no_i_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_ext_no_i_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const uint* diagIdx [[buffer(5)]], device const uint* ProwPtr [[buffer(6)]], device const uint* nRows [[buffer(7)]], device uint* PcolIdx [[buffer(8)]], device float* Pvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_type<float>(0x3f800000u);
    } else {
      if (mark == -1) {
        int iDiagPos = ((int)(diagIdx[row]));
        float diagI = values[iDiagPos];
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int mW = cfMarking[colW];
            if (mW == 1) {
              float aij = values[kw];
              float w = (-(aij / diagI));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
        for (int jf = 0; jf < (int)((re - rs)); ++jf) {
          int kf = (rs + jf);
          uint isStrF = strongMask[kf];
          if (isStrF == 1u) {
            int nbrF = ((int)(colIdx[kf]));
            int mF = cfMarking[nbrF];
            if (mF == -1) {
              float aij = values[kf];
              int jDiagPos = ((int)(diagIdx[nbrF]));
              float diagJ = values[jDiagPos];
              int rs2 = ((int)(rowPtr[nbrF]));
              int re2 = ((int)(rowPtr[(nbrF + 1)]));
              for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                int k2 = (rs2 + j2);
                uint isStr2 = strongMask[k2];
                if (isStr2 == 1u) {
                  int nbr2 = ((int)(colIdx[k2]));
                  int m2 = cfMarking[nbr2];
                  if (m2 == 1) {
                    float ajk = values[k2];
                    float prod = (aij * (ajk / diagJ));
                    float wff = (-(prod / diagI));
                    PcolIdx[(pStart + pos)] = ((uint)(nbr2));
                    Pvalues[(pStart + pos)] = wff;
                    pos = (pos + 1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_f_f_weights_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_f_f_weights_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const int* cfMarking [[buffer(4)]], device const float* diag [[buffer(5)]], device const uint* ProwPtr [[buffer(6)]], device const uint* PcolIdx [[buffer(7)]], device const uint* nRows [[buffer(8)]], device float* Pvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == -1) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      float diagVal = diag[row];
      int pStart = ((int)(ProwPtr[row]));
      int pos = 0;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == 1u) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            float aij = values[k];
            float numer = aij;
            for (int ff = 0; ff < (int)((re - rs)); ++ff) {
              int kf = (rs + ff);
              uint isStrF = strongMask[kf];
              if (isStrF == 1u) {
                int colF = ((int)(colIdx[kf]));
                int mF = cfMarking[colF];
                if (mF == -1) {
                  float aik = values[kf];
                  float diagK = diag[colF];
                  int rsK = ((int)(rowPtr[colF]));
                  int reK = ((int)(rowPtr[(colF + 1)]));
                  float akj = as_type<float>(0x00000000u);
                  for (int kn = 0; kn < (int)((reK - rsK)); ++kn) {
                    int knIdx = (rsK + kn);
                    int knCol = ((int)(colIdx[knIdx]));
                    if (knCol == col) {
                      akj = values[knIdx];
                    }
                  }
                  numer = (numer + (aik * (akj / diagK)));
                }
              }
            }
            float w = (-(numer / diagVal));
            Pvalues[(pStart + pos)] = w;
            pos = (pos + 1);
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_symbolic_row_nnz_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_symbolic_row_nnz_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* ProwNnz [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      ProwNnz[row] = 1u;
    } else {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      uint nnz = 0u;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == 1u) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            nnz = (nnz + 1u);
          }
        }
      }
      ProwNnz[row] = nnz;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_interpolate_symbolic_row_nnz_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_interpolate_symbolic_row_nnz_f64(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* ProwNnz [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 1) {
      ProwNnz[row] = 1u;
    } else {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      uint nnz = 0u;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == 1u) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            nnz = (nnz + 1u);
          }
        }
      }
      ProwNnz[row] = nnz;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_coarse_to_fine_map = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_build_coarse_to_fine_map(device const int* cfMarking [[buffer(0)]], device const uint* coarseOffset [[buffer(1)]], device const uint* nRows [[buffer(2)]], device uint* coarseToFine [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int mark = cfMarking[gid];
    if (mark == 1) {
      uint cIdx = coarseOffset[gid];
      coarseToFine[cIdx] = ((uint)(gid));
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_fine_to_coarse_map_count = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_build_fine_to_coarse_map_count(device const int* cfMarking [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* coarseFlag [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int mark = cfMarking[gid];
    coarseFlag[gid] = (((mark == 1)) ? (1u) : (0u));
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_fine_to_coarse_map = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_build_fine_to_coarse_map(device const int* cfMarking [[buffer(0)]], device const uint* coarseOffset [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* fineToCoarse [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int mark = cfMarking[gid];
    if (mark == 1) {
      uint cIdx = coarseOffset[gid];
      fineToCoarse[gid] = ((int)(cIdx));
    } else {
      fineToCoarse[gid] = -1;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_em_initial_prolongation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_em_initial_prolongation_f32(device const int* aggregateId [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* ProwPtr [[buffer(2)]], device uint* PcolIdx [[buffer(3)]], device float* Pvalues [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int aggId = aggregateId[gid];
    ProwPtr[gid] = ((uint)(gid));
    PcolIdx[gid] = ((uint)(aggId));
    Pvalues[gid] = as_type<float>(0x3f800000u);
  }
  if (gid == N) {
    ProwPtr[N] = ((uint)(N));
  }
}

)kernel";
const char* k_metal_navatala_sparse_em_energy_gradient_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_em_energy_gradient_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const uint* ProwPtr [[buffer(3)]], device const uint* PcolIdx [[buffer(4)]], device const float* Pvalues [[buffer(5)]], device const uint* nRows [[buffer(6)]], device float* gradient [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(ArowPtr[gid]));
    int re = ((int)(ArowPtr[(gid + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(AcolIdx[k]));
      float a = Avalues[k];
      int pColRs = ((int)(ProwPtr[col]));
      int pColRe = ((int)(ProwPtr[(col + 1)]));
      for (int jj = 0; jj < (int)((pColRe - pColRs)); ++jj) {
        int pk = (pColRs + jj);
        float pv = Pvalues[pk];
        sum = (sum + (a * pv));
      }
    }
    gradient[gid] = sum;
  }
}

)kernel";
const char* k_metal_navatala_sparse_em_update_prolongation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_em_update_prolongation_f32(device const float* gradient [[buffer(0)]], device const float* stepSize [[buffer(1)]], device const uint* nnzP [[buffer(2)]], device float* Pvalues [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnzP[0]));
  if (gid < N) {
    float g = gradient[gid];
    float step = stepSize[0];
    float p = Pvalues[gid];
    Pvalues[gid] = (p - (step * g));
  }
}

)kernel";
const char* k_metal_navatala_sparse_em_normalize_p_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_em_normalize_p_f32(device const uint* ProwPtr [[buffer(0)]], device const uint* nRows [[buffer(1)]], device float* Pvalues [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(ProwPtr[gid]));
    int re = ((int)(ProwPtr[(gid + 1)]));
    float rowSum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      rowSum = (rowSum + Pvalues[k]);
    }
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float v = Pvalues[k2];
      Pvalues[k2] = (v / rowSum);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_em_energy_eval_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_em_energy_eval_partials_f32(device const float* APvalues [[buffer(0)]], device const float* Pvalues [[buffer(1)]], device const uint* nnzP [[buffer(2)]], device float* numPartials [[buffer(3)]], device float* denPartials [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nnzP[0]));
  if (gid < N) {
    float ap = APvalues[gid];
    float p = Pvalues[gid];
    numPartials[gid] = (p * ap);
    denPartials[gid] = (p * p);
  }
}

)kernel";
const char* k_metal_navatala_sparse_parallel_greedy_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_parallel_greedy_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* seed [[buffer(3)]], device int* colors [[buffer(4)]], device uint* nColors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    if (myColor < 0) {
      int seedI = ((int)(seed[0]));
      int myHash = (((((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) ^ 3345072700) + ((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) ^ 3345072700) + ((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) >> 7)) << 5));
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int isMax = 1;
      int usedMask = 0;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if (col != gid) {
          int cc = colors[col];
          if (cc >= 0) {
            usedMask = (usedMask | (1 << cc));
          } else {
            int nHash = (((((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) ^ 3345072700) + ((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) >> 7)) + 374761393) ^ ((((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) ^ 3345072700) + ((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) >> 7)) << 5));
            if (nHash >= myHash) {
              isMax = 0;
            }
          }
        }
      }
      if (isMax == 1) {
        int myC = 0;
        for (int c = 0; c < (int)(32); ++c) {
          if ((usedMask & (1 << myC)) == 0) {
          } else {
            myC = (myC + 1);
          }
        }
        colors[gid] = myC;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_min_max_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_min_max_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* colors [[buffer(3)]], device uint* nColors [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    if (myColor < 0) {
      int myHash = (((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) << 5));
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int isMax = 1;
      int isMin = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if ((col != gid) && (col < N)) {
          int cc = colors[col];
          if (cc < 0) {
            int nHash = (((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) << 5));
            if (nHash >= myHash) {
              isMax = 0;
            }
            if (nHash <= myHash) {
              isMin = 0;
            }
          }
        }
      }
      int round = ((int)(nColors[0]));
      if (isMax == 1) {
        colors[gid] = (2 * round);
      } else {
        if (isMin == 1) {
          colors[gid] = ((2 * round) + 1);
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_min_max2_ring_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_min_max2_ring_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* colors [[buffer(3)]], device uint* nColors [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    if (myColor < 0) {
      int myHash = (((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) << 5));
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int isMax = 1;
      int isMin = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if ((col != gid) && (col < N)) {
          int cc = colors[col];
          if (cc < 0) {
            int nHash = (((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) << 5));
            if (nHash >= myHash) {
              isMax = 0;
            }
            if (nHash <= myHash) {
              isMin = 0;
            }
          }
          int rs2 = ((int)(rowPtr[col]));
          int re2 = ((int)(rowPtr[(col + 1)]));
          for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
            int k2 = (rs2 + j2);
            int col2 = ((int)(colIdx[k2]));
            if ((col2 != gid) && (col2 < N)) {
              int cc2 = colors[col2];
              if (cc2 < 0) {
                int n2Hash = (((((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) ^ 3345072700) + ((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) ^ 3345072700) + ((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) >> 7)) << 5));
                if (n2Hash >= myHash) {
                  isMax = 0;
                }
                if (n2Hash <= myHash) {
                  isMin = 0;
                }
              }
            }
          }
        }
      }
      int round = ((int)(nColors[0]));
      if (isMax == 1) {
        colors[gid] = (2 * round);
      } else {
        if (isMin == 1) {
          colors[gid] = ((2 * round) + 1);
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_multi_hash_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multi_hash_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* nHashFunctions [[buffer(3)]], device int* colors [[buffer(4)]], device uint* nColors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    if (myColor < 0) {
      int K = ((int)(nHashFunctions[0]));
      int h0 = (((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) << 5));
      int h1 = (((((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) ^ 3345072700) + ((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) ^ 3345072700) + ((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) >> 7)) << 5));
      int h2 = (((((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) ^ 3345072700) + ((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) ^ 3345072700) + ((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) >> 7)) << 5));
      int h3 = (((((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) ^ 3345072700) + ((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) ^ 3345072700) + ((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) >> 7)) << 5));
      int notMin = 0;
      int notMax = 0;
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if ((col != gid) && (col < N)) {
          int cc = colors[col];
          if (cc < 0) {
            int j0 = (((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) << 5));
            if (j0 >= h0) {
              notMax = (notMax | 1);
            }
            if (j0 <= h0) {
              notMin = (notMin | 1);
            }
            if (K > 1) {
              int j1 = (((((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) ^ 3345072700) + ((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) ^ 3345072700) + ((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) >> 7)) << 5));
              if (j1 >= h1) {
                notMax = (notMax | 2);
              }
              if (j1 <= h1) {
                notMin = (notMin | 2);
              }
            }
            if (K > 2) {
              int j2 = (((((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) ^ 3345072700) + ((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) ^ 3345072700) + ((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) >> 7)) << 5));
              if (j2 >= h2) {
                notMax = (notMax | 4);
              }
              if (j2 <= h2) {
                notMin = (notMin | 4);
              }
            }
            if (K > 3) {
              int j3 = (((((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) ^ 3345072700) + ((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) ^ 3345072700) + ((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) >> 7)) << 5));
              if (j3 >= h3) {
                notMax = (notMax | 8);
              }
              if (j3 <= h3) {
                notMin = (notMin | 8);
              }
            }
          }
        }
      }
      int picked = -1;
      if ((notMin & 1) == 0) {
        picked = 0;
      }
      if ((picked == -1) && ((notMax & 1) == 0)) {
        picked = 1;
      }
      if ((picked == -1) && (K > 1)) {
        if ((notMin & 2) == 0) {
          picked = 2;
        }
        if ((picked == -1) && ((notMax & 2) == 0)) {
          picked = 3;
        }
      }
      if ((picked == -1) && (K > 2)) {
        if ((notMin & 4) == 0) {
          picked = 4;
        }
        if ((picked == -1) && ((notMax & 4) == 0)) {
          picked = 5;
        }
      }
      if ((picked == -1) && (K > 3)) {
        if ((notMin & 8) == 0) {
          picked = 6;
        }
        if ((picked == -1) && ((notMax & 8) == 0)) {
          picked = 7;
        }
      }
      if (picked >= 0) {
        colors[gid] = picked;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_build_color_offsets = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_build_color_offsets(device const int* colors [[buffer(0)]], device const uint* nRows [[buffer(1)]], device const uint* nColors [[buffer(2)]], device atomic_uint* colorOffsets [[buffer(3)]], device uint* reorderPerm [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int c = colors[gid];
    uint pos = atomic_fetch_add_explicit(((device atomic_uint*)(&(colorOffsets[c]))), 1u, memory_order_relaxed);
    reorderPerm[((int)(pos))] = ((uint)(gid));
  }
}

)kernel";
const char* k_metal_navatala_sparse_reorder_c_s_r_by_color = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_reorder_c_s_r_by_color(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* reorderPerm [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* newRowPtr [[buffer(5)]], device uint* newColIdx [[buffer(6)]], device float* newValues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int origRow = ((int)(reorderPerm[gid]));
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int nnz = (re - rs);
    newRowPtr[gid] = ((uint)(nnz));
    for (int j = 0; j < (int)(nnz); ++j) {
      int k = (rs + j);
      newColIdx[(gid + j)] = colIdx[k];
      newValues[(gid + j)] = values[k];
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_round_robin_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_round_robin_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* nTargetColors [[buffer(3)]], device int* colors [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  int nC = ((int)(nTargetColors[0]));
  if (gid < N) {
    colors[gid] = (gid % nC);
  }
}

)kernel";
const char* k_metal_navatala_sparse_bfs_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_bfs_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* startRow [[buffer(3)]], device int* colors [[buffer(4)]], device uint* nColors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int minColor = 0;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        int cc = colors[col];
        if (cc == minColor) {
          minColor = (minColor + 1);
        }
      }
    }
    colors[gid] = minColor;
  }
}

)kernel";
const char* k_metal_navatala_sparse_locally_downwind_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_locally_downwind_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device int* colors [[buffer(4)]], device uint* nColors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int minColor = 0;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        int cc = colors[col];
        if (cc == minColor) {
          minColor = (minColor + 1);
        }
      }
    }
    colors[gid] = minColor;
  }
}

)kernel";
const char* k_metal_navatala_sparse_greedy_recolor_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_greedy_recolor_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* colors [[buffer(3)]], device uint* nColors [[buffer(4)]], device uint* nConflicts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int minColor = 0;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        int cc = colors[col];
        if (cc == minColor) {
          minColor = (minColor + 1);
        }
      }
    }
    colors[gid] = minColor;
  }
}

)kernel";
const char* k_metal_navatala_sparse_greedy_recolor_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_greedy_recolor_f64(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* colors [[buffer(3)]], device uint* nColors [[buffer(4)]], device uint* nConflicts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int minColor = 0;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        int cc = colors[col];
        if (cc == minColor) {
          minColor = (minColor + 1);
        }
      }
    }
    colors[gid] = minColor;
  }
}

)kernel";
const char* k_metal_navatala_sparse_validate_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_validate_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const int* colors [[buffer(2)]], device const uint* nRows [[buffer(3)]], device atomic_uint* nConflicts [[buffer(4)]], device uint* conflictFlags [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    uint conflict = 0u;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      int cc = colors[col];
      if ((col != gid) && (cc == myColor)) {
        conflict = 1u;
      }
    }
    conflictFlags[gid] = conflict;
    if (conflict == 1u) {
      uint _aod1 = atomic_fetch_add_explicit(((device atomic_uint*)(&(nConflicts[0]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_color_histogram = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_color_histogram(device const int* colors [[buffer(0)]], device const uint* nRows [[buffer(1)]], device const uint* nColors [[buffer(2)]], device atomic_uint* histogram [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int c = colors[gid];
    if (c >= 0) {
      uint _aod2 = atomic_fetch_add_explicit(((device atomic_uint*)(&(histogram[c]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_uniform_random_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_uniform_random_coloring(device const uint* nRows [[buffer(0)]], device const uint* nTargetColors [[buffer(1)]], device const uint* seed [[buffer(2)]], device int* colors [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  int nC = ((int)(nTargetColors[0]));
  if (gid < N) {
    colors[gid] = (gid % nC);
  }
}

)kernel";
const char* k_metal_navatala_sparse_saturation_coloring = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_saturation_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* colors [[buffer(3)]], device uint* nColors [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int minColor = 0;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != gid) {
        int cc = colors[col];
        if (cc == minColor) {
          minColor = (minColor + 1);
        }
      }
    }
    colors[gid] = minColor;
  }
}

)kernel";
const char* k_metal_navatala_sparse_reorder_vector_by_color_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_reorder_vector_by_color_f32(device const float* x [[buffer(0)]], device const uint* reorderPerm [[buffer(1)]], device const uint* n [[buffer(2)]], device float* xReordered [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    int origIdx = ((int)(reorderPerm[gid]));
    xReordered[gid] = x[origIdx];
  }
}

)kernel";
const char* k_metal_navatala_sparse_diag_inv_sqrt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_diag_inv_sqrt_f32(device const float* diag [[buffer(0)]], device const uint* n [[buffer(1)]], device float* dinvsqrt [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float d = diag[gid];
    dinvsqrt[gid] = (as_type<float>(0x3f800000u) / sqrt(d));
  }
}

)kernel";
const char* k_metal_navatala_sparse_apply_diag_sym_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_apply_diag_sym_scale_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* dinvsqrt [[buffer(2)]], device const uint* nRows [[buffer(3)]], device float* values [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float di = dinvsqrt[row];
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float dj = dinvsqrt[col];
      float v = values[k];
      values[k] = (v * (di * dj));
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_binorm_row_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_binorm_row_scale_f32(device const uint* rowPtr [[buffer(0)]], device const float* values [[buffer(1)]], device const uint* nRows [[buffer(2)]], device float* rowScale [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float v = values[k];
      sum = (sum + abs(v));
    }
    rowScale[row] = (as_type<float>(0x3f800000u) / sqrt(sum));
  }
}

)kernel";
const char* k_metal_navatala_sparse_shifted_sp_m_v_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_shifted_sp_m_v_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* sigma [[buffer(4)]], device const uint* nRows [[buffer(5)]], device float* y [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float xj = x[col];
      acc = (acc + (a * xj));
    }
    float s = sigma[0];
    float xi = x[row];
    y[row] = (acc - (s * xi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_deflated_project_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_deflated_project_f32(device const float* x [[buffer(0)]], device const float* Z [[buffer(1)]], device const float* ZtX [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* w [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float xi = x[gid];
    float proj = as_type<float>(0x00000000u);
    int kVal = ((int)(k[0]));
    for (int j = 0; j < (int)(kVal); ++j) {
      float zVal = Z[((j * N) + gid)];
      float zx = ZtX[j];
      proj = (proj + (zVal * zx));
    }
    w[gid] = (xi - proj);
  }
}

)kernel";
const char* k_metal_navatala_sparse_page_rank_iter_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_page_rank_iter_f32(device const float* Ax [[buffer(0)]], device const float* damping [[buffer(1)]], device const uint* n [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float axi = Ax[gid];
    float d = damping[0];
    float nInv = (as_type<float>(0x3f800000u) / ((float)(N)));
    y[gid] = (((as_type<float>(0x3f800000u) - d) * nInv) + (d * axi));
  }
}

)kernel";
const char* k_metal_navatala_sparse_w_cycle_weight_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_w_cycle_weight_partials_f32(device const float* r [[buffer(0)]], device const float* e [[buffer(1)]], device const float* Ae [[buffer(2)]], device const uint* n [[buffer(3)]], device float* rePartials [[buffer(4)]], device float* AeePartials [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float ri = r[gid];
    float ei = e[gid];
    float aei = Ae[gid];
    rePartials[gid] = (ri * ei);
    AeePartials[gid] = (aei * ei);
  }
}

)kernel";
const char* k_metal_navatala_sparse_convergence_factor_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_convergence_factor_f32(device const float* normCurrent [[buffer(0)]], device const float* normPrevious [[buffer(1)]], device float* rho [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float cur = normCurrent[0];
  float prev = normPrevious[0];
  rho[0] = (cur / prev);
}

)kernel";
const char* k_metal_navatala_sparse_block_gram_schmidt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_block_gram_schmidt_f32(device const float* X [[buffer(0)]], device const float* coeffs [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* j [[buffer(3)]], device const uint* k [[buffer(4)]], device float* Xj [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    int jVal = ((int)(j[0]));
    float xij = X[((jVal * N) + gid)];
    float v = xij;
    int kVal = ((int)(k[0]));
    for (int ii = 0; ii < (int)(jVal); ++ii) {
      float dot = coeffs[ii];
      float q = X[((ii * N) + gid)];
      v = (v - (dot * q));
    }
    Xj[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_sparse_small_dense_q_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_small_dense_q_r_f32(device const uint* m [[buffer(0)]], device const uint* nCol [[buffer(1)]], device float* A [[buffer(2)]], device float* tau [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int mVal = ((int)(m[0]));
  int nVal = ((int)(nCol[0]));
  for (int k = 0; k < (int)(nVal); ++k) {
    float norm = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)((mVal - k)); ++i) {
      float aij = A[((k * mVal) + (k + i))];
      norm = (norm + (aij * aij));
    }
    float diag = A[((k * mVal) + k)];
    float sigma = (((diag >= as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0xbf800000u)));
    float alpha = (sigma * sqrt(norm));
    tau[k] = (as_type<float>(0x40000000u) / (alpha * (alpha + diag)));
    A[((k * mVal) + k)] = (diag + alpha);
  }
}

)kernel";
const char* k_metal_navatala_sparse_lanczos_step_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_lanczos_step_f32(device const float* Av [[buffer(0)]], device const float* vCurrent [[buffer(1)]], device const float* vPrevious [[buffer(2)]], device const float* alpha [[buffer(3)]], device const float* beta [[buffer(4)]], device const uint* n [[buffer(5)]], device float* vNext [[buffer(6)]], device float* betaNext [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float avi = Av[gid];
    float vc = vCurrent[gid];
    float vp = vPrevious[gid];
    float a = alpha[0];
    float b = beta[0];
    float w = ((avi - (a * vc)) - (b * vp));
    vNext[gid] = w;
    atomic_fetch_add_explicit((device atomic_float*)(&betaNext[0]), (w * w), memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_arnoldi_step_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_arnoldi_step_f32(device const float* Av [[buffer(0)]], device const float* V [[buffer(1)]], device const float* hCol [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* j [[buffer(4)]], device float* vNext [[buffer(5)]], device float* hNextJ [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float avi = Av[gid];
    float w = avi;
    int jVal = ((int)(j[0]));
    for (int ii = 0; ii < (int)((jVal + 1)); ++ii) {
      float h = hCol[ii];
      float vi = V[((ii * N) + gid)];
      w = (w - (h * vi));
    }
    float wFinal = w;
    vNext[gid] = wFinal;
    atomic_fetch_add_explicit((device atomic_float*)(&hNextJ[0]), (wFinal * wFinal), memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_lobpcg_rayleigh_ritz_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_lobpcg_rayleigh_ritz_f32(device const float* X [[buffer(0)]], device const float* AX [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device float* XtAX [[buffer(4)]], device float* XtX [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  if (gid < (kVal * kVal)) {
    int row = (gid / kVal);
    int col = (gid % kVal);
    float dotAX = as_type<float>(0x00000000u);
    float dotXX = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(N); ++i) {
      float xi = X[((col * N) + i)];
      float axi = AX[((row * N) + i)];
      dotAX = (dotAX + (xi * axi));
      float xj = X[((row * N) + i)];
      dotXX = (dotXX + (xi * xj));
    }
    XtAX[gid] = dotAX;
    XtX[gid] = dotXX;
  }
}

)kernel";
const char* k_metal_navatala_sparse_lobpcg_residual_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_lobpcg_residual_f32(device const float* AX [[buffer(0)]], device const float* X [[buffer(1)]], device const float* lambdas [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* R [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  int total = (N * kVal);
  if (gid < total) {
    int vecIdx = (gid / N);
    int elemIdx = (gid % N);
    float ax = AX[gid];
    float x = X[gid];
    float lam = lambdas[vecIdx];
    R[gid] = (ax - (lam * x));
  }
}

)kernel";
const char* k_metal_navatala_sparse_power_iteration_normalize_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_power_iteration_normalize_f32(device const float* Av [[buffer(0)]], device const float* norm [[buffer(1)]], device const uint* n [[buffer(2)]], device float* v [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float avi = Av[gid];
    float nrm = norm[0];
    v[gid] = (avi / nrm);
  }
}

)kernel";
const char* k_metal_navatala_sparse_rayleigh_quotient_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_rayleigh_quotient_partials_f32(device const float* v [[buffer(0)]], device const float* Av [[buffer(1)]], device const uint* n [[buffer(2)]], device float* vAvPartials [[buffer(3)]], device float* vvPartials [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float vi = v[gid];
    float avi = Av[gid];
    vAvPartials[gid] = (vi * avi);
    vvPartials[gid] = (vi * vi);
  }
}

)kernel";
const char* k_metal_navatala_sparse_jacobi_davidson_correction_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_jacobi_davidson_correction_f32(device const float* r [[buffer(0)]], device const float* v [[buffer(1)]], device const float* theta [[buffer(2)]], device const uint* n [[buffer(3)]], device float* t [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float ri = r[gid];
    float th = theta[0];
    float negR = (-ri);
    float absTh = abs(th);
    float safeTh = (((absTh > as_type<float>(0x0da24260u))) ? (th) : (as_type<float>(0x3f800000u)));
    bool useDiag = (absTh > as_type<float>(0x0da24260u));
    float ti = ((useDiag) ? ((negR / safeTh)) : (negR));
    t[gid] = ti;
  }
}

)kernel";
const char* k_metal_navatala_sparse_jd_subspace_expand_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_jd_subspace_expand_f32(device const float* V [[buffer(0)]], device const float* t [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device float* vNew [[buffer(4)]], device float* normNew [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float ti = t[gid];
    float w = ti;
    int kVal = ((int)(k[0]));
    for (int j = 0; j < (int)(kVal); ++j) {
      float vi = V[((j * N) + gid)];
      w = (w - ((ti * vi) * vi));
    }
    vNew[gid] = w;
  }
}

)kernel";
const char* k_metal_navatala_sparse_subspace_orthogonalize_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_subspace_orthogonalize_f32(device const uint* n [[buffer(0)]], device const uint* k [[buffer(1)]], device float* X [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  for (int j = 0; j < (int)(kVal); ++j) {
    if (gid < N) {
      float xij = X[((j * N) + gid)];
      float v = xij;
      for (int ii = 0; ii < (int)(j); ++ii) {
        float q = X[((ii * N) + gid)];
        v = (v - (q * q));
      }
      X[((j * N) + gid)] = v;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_subspace_convergence_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_subspace_convergence_f32(device const float* AX [[buffer(0)]], device const float* X [[buffer(1)]], device const float* lambdas [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const float* tolerance [[buffer(5)]], device float* residNorms [[buffer(6)]], device atomic_uint* nConverged [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  if (gid < kVal) {
    float norm2 = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(N); ++i) {
      float ax = AX[((gid * N) + i)];
      float x = X[((gid * N) + i)];
      float lam = lambdas[gid];
      norm2 = (norm2 + ((ax - (lam * x)) * (ax - (lam * x))));
    }
    residNorms[gid] = norm2;
    float tol = tolerance[0];
    if (norm2 < (tol * tol)) {
      uint _aod1 = atomic_fetch_add_explicit(((device atomic_uint*)(&(nConverged[0]))), 1u, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_block_sp_m_v_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_block_sp_m_v_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* X [[buffer(3)]], device const uint* nRows [[buffer(4)]], device const uint* k [[buffer(5)]], device float* Y [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int total = (((int)(nRows[0])) * ((int)(k[0])));
  int nR = ((int)(nRows[0]));
  if (gid < total) {
    int row = (gid % nR);
    int vecIdx = (gid / nR);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k2 = (rs + j);
      int col = ((int)(colIdx[k2]));
      float a = values[k2];
      float x = X[((vecIdx * nR) + col)];
      sum = (sum + (a * x));
    }
    Y[gid] = sum;
  }
}

)kernel";
const char* k_metal_navatala_sparse_multi_vec_norm_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multi_vec_norm_partials_f32(device const float* X [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* partials [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  int total = (N * kVal);
  if (gid < total) {
    float x = X[gid];
    partials[gid] = (x * x);
  }
}

)kernel";
const char* k_metal_navatala_sparse_multi_vec_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_multi_vec_scale_f32(device const float* norms [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* X [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  int total = (N * kVal);
  if (gid < total) {
    int vecIdx = (gid / N);
    float x = X[gid];
    float nrm = norms[vecIdx];
    X[gid] = (x / nrm);
  }
}

)kernel";
const char* k_metal_navatala_sparse_dense_sym_eig_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dense_sym_eig_f32(device const float* M [[buffer(0)]], device const uint* k [[buffer(1)]], device float* eigenvalues [[buffer(2)]], device float* eigenvectors [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int kVal = ((int)(k[0]));
  for (int i = 0; i < (int)(kVal); ++i) {
    for (int j = 0; j < (int)(kVal); ++j) {
      float val = M[((i * kVal) + j)];
      float isOne = (((i == j)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
      eigenvalues[i] = M[((i * kVal) + i)];
      eigenvectors[((i * kVal) + j)] = isOne;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_convergence_relative_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_convergence_relative_max_f32(device const float* residualNorm [[buffer(0)]], device const float* maxNorm [[buffer(1)]], device const float* tolerance [[buffer(2)]], device uint* converged [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float mx = maxNorm[0];
  float tol = tolerance[0];
  float ratio = (norm / mx);
  uint flag = (((ratio < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_metal_navatala_sparse_convergence_combined_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_convergence_combined_f32(device const float* residualNorm [[buffer(0)]], device const float* initialNorm [[buffer(1)]], device const float* absTol [[buffer(2)]], device const float* relTol [[buffer(3)]], device uint* converged [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float init = initialNorm[0];
  float at = absTol[0];
  float rt = relTol[0];
  uint absConv = (((norm < at)) ? (1u) : (0u));
  float ratio = (norm / init);
  uint relConv = (((ratio < rt)) ? (1u) : (0u));
  uint flag = ((((absConv == 1u) || (relConv == 1u))) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_metal_navatala_sparse_convergence_divergence_check_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_convergence_divergence_check_f32(device const float* residualNorm [[buffer(0)]], device const float* prevNorm [[buffer(1)]], device const uint* consecutiveIncreases [[buffer(2)]], device uint* diverged [[buffer(3)]], device uint* newConsecutiveIncreases [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float prev = prevNorm[0];
  uint ci = consecutiveIncreases[0];
  uint increased = (((norm > prev)) ? (1u) : (0u));
  uint newCi = (((increased == 1u)) ? ((ci + 1u)) : (0u));
  uint div = (((newCi >= 3u)) ? (1u) : (0u));
  diverged[0] = div;
  newConsecutiveIncreases[0] = newCi;
}

)kernel";
const char* k_metal_navatala_sparse_update_max_norm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_update_max_norm_f32(device const float* residualNorm [[buffer(0)]], device float* maxNorm [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float mx = maxNorm[0];
  float newMx = (((norm > mx)) ? (norm) : (mx));
  maxNorm[0] = newMx;
}

)kernel";
const char* k_metal_navatala_sparse_amg_level_metrics_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_amg_level_metrics_f32(device const uint* levelNRows [[buffer(0)]], device const uint* levelNnz [[buffer(1)]], device const uint* nLevels [[buffer(2)]], device float* gridComplexity [[buffer(3)]], device float* operatorComplexity [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int nL = ((int)(nLevels[0]));
  float rows0 = ((float)(levelNRows[0]));
  float nnz0 = ((float)(levelNnz[0]));
  float totalRows = as_type<float>(0x00000000u);
  float totalNnz = as_type<float>(0x00000000u);
  for (int i = 0; i < (int)(nL); ++i) {
    float ri = ((float)(levelNRows[i]));
    float ni = ((float)(levelNnz[i]));
    totalRows = (totalRows + ri);
    totalNnz = (totalNnz + ni);
  }
  gridComplexity[0] = (totalRows / rows0);
  operatorComplexity[0] = (totalNnz / nnz0);
}

)kernel";
const char* k_metal_navatala_sparse_amg_coarse_size_check_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_amg_coarse_size_check_f32(device const uint* coarseNRows [[buffer(0)]], device const uint* denseLUThreshold [[buffer(1)]], device uint* useDenseLU [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint cn = coarseNRows[0];
  uint th = denseLUThreshold[0];
  uint flag = (((cn < th)) ? (1u) : (0u));
  useDenseLU[0] = flag;
}

)kernel";
const char* k_metal_navatala_sparse_amg_cycle_counter_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_amg_cycle_counter_f32(device const float* residualNorm [[buffer(0)]], device const float* tolerance [[buffer(1)]], device const uint* maxCycles [[buffer(2)]], device uint* cycleCount [[buffer(3)]], device uint* converged [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float tol = tolerance[0];
  uint cc = cycleCount[0];
  uint mc = maxCycles[0];
  uint newCc = (cc + 1u);
  uint conv = (((norm < tol)) ? (1u) : (0u));
  uint maxReached = (((newCc >= mc)) ? (1u) : (0u));
  uint flag = ((((conv == 1u) || (maxReached == 1u))) ? (1u) : (0u));
  cycleCount[0] = newCc;
  converged[0] = flag;
}

)kernel";
const char* k_metal_navatala_sparse_amg_validate_coarsening_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_amg_validate_coarsening_f32(device const uint* fineNRows [[buffer(0)]], device const uint* coarseNRows [[buffer(1)]], device const float* minRatio [[buffer(2)]], device const float* maxRatio [[buffer(3)]], device uint* acceptable [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float fine = ((float)(fineNRows[0]));
  float coarse = ((float)(coarseNRows[0]));
  float ratio = (coarse / fine);
  float minR = minRatio[0];
  float maxR = maxRatio[0];
  uint ok = ((((ratio >= minR) && (ratio <= maxR))) ? (1u) : (0u));
  acceptable[0] = ok;
}

)kernel";
const char* k_metal_navatala_sparse_halo_pack_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_halo_pack_f32(device const float* x [[buffer(0)]], device const uint* sendMap [[buffer(1)]], device const uint* sendCount [[buffer(2)]], device float* sendBuf [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(sendCount[0]));
  if (gid < N) {
    int idx = ((int)(sendMap[gid]));
    float val = x[idx];
    sendBuf[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_sparse_halo_unpack_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_halo_unpack_f32(device const float* recvBuf [[buffer(0)]], device const uint* recvMap [[buffer(1)]], device const uint* recvCount [[buffer(2)]], device float* xHalo [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(recvCount[0]));
  if (gid < N) {
    int idx = ((int)(recvMap[gid]));
    float val = recvBuf[gid];
    xHalo[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_sparse_halo_pack_b_s_r_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_halo_pack_b_s_r_f32(device const float* x [[buffer(0)]], device const uint* sendMap [[buffer(1)]], device const uint* sendCount [[buffer(2)]], device const uint* blockDim [[buffer(3)]], device float* sendBuf [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(sendCount[0]));
  int bd = ((int)(blockDim[0]));
  if (gid < (N * bd)) {
    int entry = (gid / bd);
    int comp = (gid % bd);
    int idx = ((((int)(sendMap[entry])) * bd) + comp);
    float val = x[idx];
    sendBuf[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_sparse_partition_count_per_rank = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_partition_count_per_rank(device const uint* partitionVector [[buffer(0)]], device const uint* nRows [[buffer(1)]], device const uint* nRanks [[buffer(2)]], device atomic_uint* countPerRank [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rank = ((int)(partitionVector[gid]));
    uint _aod1 = atomic_fetch_add_explicit(((device atomic_uint*)(&(countPerRank[rank]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_partition_reorder_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_partition_reorder_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* perm [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* newRowPtr [[buffer(5)]], device uint* newColIdx [[buffer(6)]], device float* newValues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int origRow = ((int)(perm[gid]));
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int nnz = (re - rs);
    newRowPtr[gid] = ((uint)(nnz));
    for (int j = 0; j < (int)(nnz); ++j) {
      int k = (rs + j);
      uint col = colIdx[k];
      float val = values[k];
      int off = (gid + j);
      newColIdx[off] = col;
      newValues[off] = val;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_global_to_local_index_map = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_global_to_local_index_map(device const uint* globalIndices [[buffer(0)]], device const uint* localOffset [[buffer(1)]], device const uint* nLocal [[buffer(2)]], device const uint* nIndices [[buffer(3)]], device uint* localIndices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nIndices[0]));
  if (gid < N) {
    uint gIdx = globalIndices[gid];
    uint offset = localOffset[0];
    uint nLoc = nLocal[0];
    uint localIdx = (gIdx - offset);
    bool valid = (localIdx < nLoc);
    localIndices[gid] = ((valid) ? (localIdx) : (4294967295u));
  }
}

)kernel";
const char* k_metal_navatala_sparse_distributed_sp_m_v_halo_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_distributed_sp_m_v_halo_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* xHalo [[buffer(3)]], device const uint* nRows [[buffer(4)]], device float* yHalo [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float xj = xHalo[col];
      sum = (sum + (a * xj));
    }
    yHalo[gid] = sum;
  }
}

)kernel";
const char* k_metal_navatala_sparse_merge_local_halo_result_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_merge_local_halo_result_f32(device const float* yLocal [[buffer(0)]], device const float* yHalo [[buffer(1)]], device const uint* n [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float yl = yLocal[gid];
    float yh = yHalo[gid];
    y[gid] = (yl + yh);
  }
}

)kernel";
const char* k_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ldu_to_csr_init_diagonal_count_f32(device const uint* nCells [[buffer(0)]], device uint* rowNnz [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    rowNnz[cell] = 1u;
  }
}

)kernel";
const char* k_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ldu_to_csr_init_diagonal_count_f64(device const uint* nCells [[buffer(0)]], device uint* rowNnz [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    rowNnz[cell] = 1u;
  }
}

)kernel";
const char* k_metal_navatala_sparse_ldu_to_csr_count_nnz_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_ldu_to_csr_count_nnz_f32(device const uint* owner [[buffer(0)]], device const uint* neighbour [[buffer(1)]], device const uint* nInternalFaces [[buffer(2)]], device const uint* nCells [[buffer(3)]], device atomic_uint* rowNnz [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int face = int(__gid.x);
  int nFaces = ((int)(nInternalFaces[0]));
  if (face < nFaces) {
    int own = ((int)(owner[face]));
    int nbr = ((int)(neighbour[face]));
    uint _a = atomic_fetch_add_explicit(((device atomic_uint*)(&(rowNnz[own]))), 1u, memory_order_relaxed);
    uint _b = atomic_fetch_add_explicit(((device atomic_uint*)(&(rowNnz[nbr]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_ldu_to_csr_count_nnz_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_ldu_to_csr_count_nnz_f64(device const uint* owner [[buffer(0)]], device const uint* neighbour [[buffer(1)]], device const uint* nInternalFaces [[buffer(2)]], device const uint* nCells [[buffer(3)]], device atomic_uint* rowNnz [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int face = int(__gid.x);
  int nFaces = ((int)(nInternalFaces[0]));
  if (face < nFaces) {
    int own = ((int)(owner[face]));
    int nbr = ((int)(neighbour[face]));
    uint _a = atomic_fetch_add_explicit(((device atomic_uint*)(&(rowNnz[own]))), 1u, memory_order_relaxed);
    uint _b = atomic_fetch_add_explicit(((device atomic_uint*)(&(rowNnz[nbr]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_sparse_ldu_to_csr_fill_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_ldu_to_csr_fill_f32(device const uint* owner [[buffer(0)]], device const uint* neighbour [[buffer(1)]], device const float* upper [[buffer(2)]], device const float* lower [[buffer(3)]], device const uint* rowPtr [[buffer(4)]], device const uint* nInternalFaces [[buffer(5)]], device const uint* nCells [[buffer(6)]], device uint* colInd [[buffer(7)]], device float* values [[buffer(8)]], device atomic_uint* insertIdx [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int face = int(__gid.x);
  int nFaces = ((int)(nInternalFaces[0]));
  if (face < nFaces) {
    int own = ((int)(owner[face]));
    int nbr = ((int)(neighbour[face]));
    uint posOwn = atomic_fetch_add_explicit(((device atomic_uint*)(&(insertIdx[own]))), 1u, memory_order_relaxed);
    int offOwn = (((int)(rowPtr[own])) + ((int)(posOwn)));
    colInd[offOwn] = neighbour[face];
    values[offOwn] = upper[face];
    uint posNbr = atomic_fetch_add_explicit(((device atomic_uint*)(&(insertIdx[nbr]))), 1u, memory_order_relaxed);
    int offNbr = (((int)(rowPtr[nbr])) + ((int)(posNbr)));
    colInd[offNbr] = owner[face];
    values[offNbr] = lower[face];
  }
}

)kernel";
const char* k_metal_navatala_sparse_ldu_diagonal_fill_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_ldu_diagonal_fill_f32(device const float* diagonal [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* nCells [[buffer(2)]], device uint* colInd [[buffer(3)]], device float* values [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (row < N) {
    int rEnd = ((int)(rowPtr[(row + 1)]));
    int dPos = (rEnd - 1);
    colInd[dPos] = ((uint)(row));
    values[dPos] = diagonal[row];
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_so_c_ldu_count_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_compute_so_c_ldu_count_f32(device const uint* owner [[buffer(0)]], device const float* upper [[buffer(1)]], device const float* lower [[buffer(2)]], device const uint* cellFaceOffsets [[buffer(3)]], device const uint* cellFaceIdx [[buffer(4)]], device const float* theta [[buffer(5)]], device const uint* nCells [[buffer(6)]], device uint* nnzPerRow [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    int fStart = ((int)(cellFaceOffsets[cell]));
    int fEnd = ((int)(cellFaceOffsets[(cell + 1)]));
    float maxCoeff = as_type<float>(0x00000000u);
    for (int fi = 0; fi < (int)((fEnd - fStart)); ++fi) {
      int fIdx = (fStart + fi);
      int face = ((int)(cellFaceIdx[fIdx]));
      int own = ((int)(owner[face]));
      float coeff = abs((((cell == own)) ? (upper[face]) : (lower[face])));
      if (coeff > maxCoeff) {
        maxCoeff = coeff;
      }
    }
    float th = theta[0];
    float threshold = (th * maxCoeff);
    int count = 0;
    for (int fi2 = 0; fi2 < (int)((fEnd - fStart)); ++fi2) {
      int fIdx2 = (fStart + fi2);
      int face2 = ((int)(cellFaceIdx[fIdx2]));
      int own2 = ((int)(owner[face2]));
      float coeff2 = abs((((cell == own2)) ? (upper[face2]) : (lower[face2])));
      if (coeff2 > threshold) {
        count = (count + 1);
      }
    }
    nnzPerRow[cell] = ((uint)(count));
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_so_c_ldu_fill_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_compute_so_c_ldu_fill_f32(device const uint* owner [[buffer(0)]], device const uint* neighbour [[buffer(1)]], device const float* upper [[buffer(2)]], device const float* lower [[buffer(3)]], device const uint* cellFaceOffsets [[buffer(4)]], device const uint* cellFaceIdx [[buffer(5)]], device const uint* strongRowPtr [[buffer(6)]], device const float* theta [[buffer(7)]], device const uint* nCells [[buffer(8)]], device uint* strongColIdx [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    int fStart = ((int)(cellFaceOffsets[cell]));
    int fEnd = ((int)(cellFaceOffsets[(cell + 1)]));
    float maxCoeff = as_type<float>(0x00000000u);
    for (int fi = 0; fi < (int)((fEnd - fStart)); ++fi) {
      int fIdx = (fStart + fi);
      int face = ((int)(cellFaceIdx[fIdx]));
      int own = ((int)(owner[face]));
      float coeff = abs((((cell == own)) ? (upper[face]) : (lower[face])));
      if (coeff > maxCoeff) {
        maxCoeff = coeff;
      }
    }
    int rowStart = ((int)(strongRowPtr[cell]));
    float th = theta[0];
    float threshold = (th * maxCoeff);
    int off = 0;
    for (int fi2 = 0; fi2 < (int)((fEnd - fStart)); ++fi2) {
      int fIdx2 = (fStart + fi2);
      int face2 = ((int)(cellFaceIdx[fIdx2]));
      int own2 = ((int)(owner[face2]));
      float coeff2 = abs((((cell == own2)) ? (upper[face2]) : (lower[face2])));
      if (coeff2 > threshold) {
        int nbr2 = ((int)((((cell == own2)) ? (neighbour[face2]) : (owner[face2]))));
        strongColIdx[(rowStart + off)] = ((uint)(nbr2));
        off = (off + 1);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_m_i_s_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_compute_m_i_s_f32(device const uint* strongRowPtr [[buffer(0)]], device const uint* strongColIdx [[buffer(1)]], device const uint* randomPriority [[buffer(2)]], device const uint* nCells [[buffer(3)]], device int* aggregateId [[buffer(4)]], device atomic_uint* nUndecided [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    int myAgg = aggregateId[cell];
    if (myAgg == -1) {
      uint myPri = randomPriority[cell];
      int rs = ((int)(strongRowPtr[cell]));
      int re = ((int)(strongRowPtr[(cell + 1)]));
      int isMax = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int nbr = ((int)(strongColIdx[k]));
        int nbrAgg = aggregateId[nbr];
        if (nbrAgg == -1) {
          uint nbrPri = randomPriority[nbr];
          if ((nbrPri > myPri) || ((nbrPri == myPri) && (nbr < cell))) {
            isMax = 0;
          }
        }
      }
      if (isMax == 1) {
        aggregateId[cell] = cell;
      } else {
        int assigned = 0;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          if (assigned == 0) {
            int k2 = (rs + j2);
            int nbr2 = ((int)(strongColIdx[k2]));
            int nbrAgg2 = aggregateId[nbr2];
            if (nbrAgg2 >= 0) {
              aggregateId[cell] = nbrAgg2;
              assigned = 1;
            }
          }
        }
        if (assigned == 0) {
          uint _u = atomic_fetch_add_explicit(((device atomic_uint*)(&(nUndecided[0]))), 1u, memory_order_relaxed);
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_compute_m_i_s_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_sparse_compute_m_i_s_f64(device const uint* strongRowPtr [[buffer(0)]], device const uint* strongColIdx [[buffer(1)]], device const uint* randomPriority [[buffer(2)]], device const uint* nCells [[buffer(3)]], device int* aggregateId [[buffer(4)]], device atomic_uint* nUndecided [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if (cell < N) {
    int myAgg = aggregateId[cell];
    if (myAgg == -1) {
      uint myPri = randomPriority[cell];
      int rs = ((int)(strongRowPtr[cell]));
      int re = ((int)(strongRowPtr[(cell + 1)]));
      int isMax = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int nbr = ((int)(strongColIdx[k]));
        int nbrAgg = aggregateId[nbr];
        if (nbrAgg == -1) {
          uint nbrPri = randomPriority[nbr];
          if ((nbrPri > myPri) || ((nbrPri == myPri) && (nbr < cell))) {
            isMax = 0;
          }
        }
      }
      if (isMax == 1) {
        aggregateId[cell] = cell;
      } else {
        int assigned = 0;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          if (assigned == 0) {
            int k2 = (rs + j2);
            int nbr2 = ((int)(strongColIdx[k2]));
            int nbrAgg2 = aggregateId[nbr2];
            if (nbrAgg2 >= 0) {
              aggregateId[cell] = nbrAgg2;
              assigned = 1;
            }
          }
        }
        if (assigned == 0) {
          uint _u = atomic_fetch_add_explicit(((device atomic_uint*)(&(nUndecided[0]))), 1u, memory_order_relaxed);
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_csr_sort_after_ldu_fill_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_csr_sort_after_ldu_fill_f32(device const uint* rowPtr [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* colInd [[buffer(2)]], device float* values [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int start = ((int)(rowPtr[row]));
    int end = ((int)(rowPtr[(row + 1)]));
    int count = (end - start);
    for (int i = 0; i < (int)(count); ++i) {
      int pos = (start + i);
      uint keyCol = colInd[pos];
      float keyVal = values[pos];
      int j = i;
      for (int __iter = 0; __iter < 64; ++__iter) {
        if (!((j > 0) && (colInd[((start + j) - 1)] > keyCol))) break;
        colInd[(start + j)] = colInd[((start + j) - 1)];
        values[(start + j)] = values[((start + j) - 1)];
        j = (j - 1);
      }
      colInd[(start + j)] = keyCol;
      values[(start + j)] = keyVal;
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_add_f32[] = {
  { "valuesA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "valuesB", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesC", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_add_f32 = {
  1,
  "navatala_sparse_sparse_add_f32",
  "metal",
  "navatala_sparse_sparse_add_f32",
  "kernel:metal:navatala_sparse_sparse_add_f32",
  "abi-r1:metal:navatala_sparse_sparse_add_f32",
  "abi-r1:metal:navatala_sparse_sparse_add_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_add_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_sub_f32[] = {
  { "valuesA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "valuesB", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesC", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_sub_f32 = {
  1,
  "navatala_sparse_sparse_sub_f32",
  "metal",
  "navatala_sparse_sparse_sub_f32",
  "kernel:metal:navatala_sparse_sparse_sub_f32",
  "abi-r1:metal:navatala_sparse_sparse_sub_f32",
  "abi-r1:metal:navatala_sparse_sparse_sub_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_sub_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_scale_f32[] = {
  { "valuesA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesB", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_scale_f32 = {
  1,
  "navatala_sparse_sparse_scale_f32",
  "metal",
  "navatala_sparse_sparse_scale_f32",
  "kernel:metal:navatala_sparse_sparse_scale_f32",
  "abi-r1:metal:navatala_sparse_sparse_scale_f32",
  "abi-r1:metal:navatala_sparse_sparse_scale_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_hadamard_f32[] = {
  { "valuesA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "valuesB", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesC", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_hadamard_f32 = {
  1,
  "navatala_sparse_sparse_hadamard_f32",
  "metal",
  "navatala_sparse_sparse_hadamard_f32",
  "kernel:metal:navatala_sparse_sparse_hadamard_f32",
  "abi-r1:metal:navatala_sparse_sparse_hadamard_f32",
  "abi-r1:metal:navatala_sparse_sparse_hadamard_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_hadamard_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_abs_f32[] = {
  { "valuesA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesB", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_abs_f32 = {
  1,
  "navatala_sparse_sparse_abs_f32",
  "metal",
  "navatala_sparse_sparse_abs_f32",
  "kernel:metal:navatala_sparse_sparse_abs_f32",
  "abi-r1:metal:navatala_sparse_sparse_abs_f32",
  "abi-r1:metal:navatala_sparse_sparse_abs_f32",
  3,
  kAbiArgs_metal_navatala_sparse_sparse_abs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_threshold_f32[] = {
  { "valuesA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "threshold", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesB", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_threshold_f32 = {
  1,
  "navatala_sparse_sparse_threshold_f32",
  "metal",
  "navatala_sparse_sparse_threshold_f32",
  "kernel:metal:navatala_sparse_sparse_threshold_f32",
  "abi-r1:metal:navatala_sparse_sparse_threshold_f32",
  "abi-r1:metal:navatala_sparse_sparse_threshold_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_threshold_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_coo_transpose_f32[] = {
  { "rowInd", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowInd_out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd_out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values_out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_coo_transpose_f32 = {
  1,
  "navatala_sparse_coo_transpose_f32",
  "metal",
  "navatala_sparse_coo_transpose_f32",
  "kernel:metal:navatala_sparse_coo_transpose_f32",
  "abi-r1:metal:navatala_sparse_coo_transpose_f32",
  "abi-r1:metal:navatala_sparse_coo_transpose_f32",
  7,
  kAbiArgs_metal_navatala_sparse_coo_transpose_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_inner_product_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_inner_product_f32 = {
  1,
  "navatala_sparse_sparse_inner_product_f32",
  "metal",
  "navatala_sparse_sparse_inner_product_f32",
  "kernel:metal:navatala_sparse_sparse_inner_product_f32",
  "abi-r1:metal:navatala_sparse_sparse_inner_product_f32",
  "abi-r1:metal:navatala_sparse_sparse_inner_product_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_l2_distance_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_l2_distance_f32 = {
  1,
  "navatala_sparse_sparse_l2_distance_f32",
  "metal",
  "navatala_sparse_sparse_l2_distance_f32",
  "kernel:metal:navatala_sparse_sparse_l2_distance_f32",
  "abi-r1:metal:navatala_sparse_sparse_l2_distance_f32",
  "abi-r1:metal:navatala_sparse_sparse_l2_distance_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_l2_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_cosine_distance_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_cosine_distance_f32 = {
  1,
  "navatala_sparse_sparse_cosine_distance_f32",
  "metal",
  "navatala_sparse_sparse_cosine_distance_f32",
  "kernel:metal:navatala_sparse_sparse_cosine_distance_f32",
  "abi-r1:metal:navatala_sparse_sparse_cosine_distance_f32",
  "abi-r1:metal:navatala_sparse_sparse_cosine_distance_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_cosine_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_jaccard_distance_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_jaccard_distance_f32 = {
  1,
  "navatala_sparse_sparse_jaccard_distance_f32",
  "metal",
  "navatala_sparse_sparse_jaccard_distance_f32",
  "kernel:metal:navatala_sparse_sparse_jaccard_distance_f32",
  "abi-r1:metal:navatala_sparse_sparse_jaccard_distance_f32",
  "abi-r1:metal:navatala_sparse_sparse_jaccard_distance_f32",
  4,
  kAbiArgs_metal_navatala_sparse_sparse_jaccard_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_jacobi_iter_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x_new", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_jacobi_iter_f32 = {
  1,
  "navatala_sparse_sparse_jacobi_iter_f32",
  "metal",
  "navatala_sparse_sparse_jacobi_iter_f32",
  "kernel:metal:navatala_sparse_sparse_jacobi_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_jacobi_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_jacobi_iter_f32",
  7,
  kAbiArgs_metal_navatala_sparse_sparse_jacobi_iter_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_gauss_seidel_iter_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x_new", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_gauss_seidel_iter_f32 = {
  1,
  "navatala_sparse_sparse_gauss_seidel_iter_f32",
  "metal",
  "navatala_sparse_sparse_gauss_seidel_iter_f32",
  "kernel:metal:navatala_sparse_sparse_gauss_seidel_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_gauss_seidel_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_gauss_seidel_iter_f32",
  7,
  kAbiArgs_metal_navatala_sparse_sparse_gauss_seidel_iter_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_cg_iter_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rTr", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x_new", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r_new", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_cg_iter_f32 = {
  1,
  "navatala_sparse_sparse_cg_iter_f32",
  "metal",
  "navatala_sparse_sparse_cg_iter_f32",
  "kernel:metal:navatala_sparse_sparse_cg_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_cg_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_cg_iter_f32",
  12,
  kAbiArgs_metal_navatala_sparse_sparse_cg_iter_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_sparse_bicgstab_iter_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "t", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x_new", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r_new", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_sparse_bicgstab_iter_f32 = {
  1,
  "navatala_sparse_sparse_bicgstab_iter_f32",
  "metal",
  "navatala_sparse_sparse_bicgstab_iter_f32",
  "kernel:metal:navatala_sparse_sparse_bicgstab_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_bicgstab_iter_f32",
  "abi-r1:metal:navatala_sparse_sparse_bicgstab_iter_f32",
  11,
  kAbiArgs_metal_navatala_sparse_sparse_bicgstab_iter_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_residual_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "residual", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_residual_f32 = {
  1,
  "navatala_sparse_compute_residual_f32",
  "metal",
  "navatala_sparse_compute_residual_f32",
  "kernel:metal:navatala_sparse_compute_residual_f32",
  "abi-r1:metal:navatala_sparse_compute_residual_f32",
  "abi-r1:metal:navatala_sparse_compute_residual_f32",
  7,
  kAbiArgs_metal_navatala_sparse_compute_residual_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_residual_norm_f32[] = {
  { "residual", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "normSq", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_residual_norm_f32 = {
  1,
  "navatala_sparse_compute_residual_norm_f32",
  "metal",
  "navatala_sparse_compute_residual_norm_f32",
  "kernel:metal:navatala_sparse_compute_residual_norm_f32",
  "abi-r1:metal:navatala_sparse_compute_residual_norm_f32",
  "abi-r1:metal:navatala_sparse_compute_residual_norm_f32",
  3,
  kAbiArgs_metal_navatala_sparse_compute_residual_norm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_spmv_row_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_spmv_row_f32 = {
  1,
  "navatala_sparse_csr_spmv_row_f32",
  "metal",
  "navatala_sparse_csr_spmv_row_f32",
  "kernel:metal:navatala_sparse_csr_spmv_row_f32",
  "abi-r1:metal:navatala_sparse_csr_spmv_row_f32",
  "abi-r1:metal:navatala_sparse_csr_spmv_row_f32",
  6,
  kAbiArgs_metal_navatala_sparse_csr_spmv_row_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_chebyshev_smooth_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "xPrev", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNext", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_chebyshev_smooth_f32 = {
  1,
  "navatala_sparse_chebyshev_smooth_f32",
  "metal",
  "navatala_sparse_chebyshev_smooth_f32",
  "kernel:metal:navatala_sparse_chebyshev_smooth_f32",
  "abi-r1:metal:navatala_sparse_chebyshev_smooth_f32",
  "abi-r1:metal:navatala_sparse_chebyshev_smooth_f32",
  11,
  kAbiArgs_metal_navatala_sparse_chebyshev_smooth_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_find_min_edge_per_component_f32[] = {
  { "srcNodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "components", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numNodes", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minEdgeIdx", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minEdgeWeight", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_find_min_edge_per_component_f32 = {
  1,
  "navatala_sparse_find_min_edge_per_component_f32",
  "metal",
  "navatala_sparse_find_min_edge_per_component_f32",
  "kernel:metal:navatala_sparse_find_min_edge_per_component_f32",
  "abi-r1:metal:navatala_sparse_find_min_edge_per_component_f32",
  "abi-r1:metal:navatala_sparse_find_min_edge_per_component_f32",
  8,
  kAbiArgs_metal_navatala_sparse_find_min_edge_per_component_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_merge_components_f32[] = {
  { "srcNodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minEdgeIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numNodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "components", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mstEdges", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_merge_components_f32 = {
  1,
  "navatala_sparse_merge_components_f32",
  "metal",
  "navatala_sparse_merge_components_f32",
  "kernel:metal:navatala_sparse_merge_components_f32",
  "abi-r1:metal:navatala_sparse_merge_components_f32",
  "abi-r1:metal:navatala_sparse_merge_components_f32",
  6,
  kAbiArgs_metal_navatala_sparse_merge_components_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_merge_components_f64[] = {
  { "srcNodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minEdgeIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numNodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "components", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mstEdges", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_merge_components_f64 = {
  1,
  "navatala_sparse_merge_components_f64",
  "metal",
  "navatala_sparse_merge_components_f64",
  "kernel:metal:navatala_sparse_merge_components_f64",
  "abi-r1:metal:navatala_sparse_merge_components_f64",
  "abi-r1:metal:navatala_sparse_merge_components_f64",
  6,
  kAbiArgs_metal_navatala_sparse_merge_components_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mst_boruvka_step_f32[] = {
  { "srcNodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numNodes", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "components", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minEdgeIdx", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minEdgeWeight", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mstEdges", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mst_boruvka_step_f32 = {
  1,
  "navatala_sparse_mst_boruvka_step_f32",
  "metal",
  "navatala_sparse_mst_boruvka_step_f32",
  "kernel:metal:navatala_sparse_mst_boruvka_step_f32",
  "abi-r1:metal:navatala_sparse_mst_boruvka_step_f32",
  "abi-r1:metal:navatala_sparse_mst_boruvka_step_f32",
  9,
  kAbiArgs_metal_navatala_sparse_mst_boruvka_step_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_sp_m_v_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_sp_m_v_f32 = {
  1,
  "navatala_sparse_csr_sp_m_v_f32",
  "metal",
  "navatala_sparse_csr_sp_m_v_f32",
  "kernel:metal:navatala_sparse_csr_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_csr_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_csr_sp_m_v_f32",
  6,
  kAbiArgs_metal_navatala_sparse_csr_sp_m_v_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_sp_m_v_scaled_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_sp_m_v_scaled_f32 = {
  1,
  "navatala_sparse_csr_sp_m_v_scaled_f32",
  "metal",
  "navatala_sparse_csr_sp_m_v_scaled_f32",
  "kernel:metal:navatala_sparse_csr_sp_m_v_scaled_f32",
  "abi-r1:metal:navatala_sparse_csr_sp_m_v_scaled_f32",
  "abi-r1:metal:navatala_sparse_csr_sp_m_v_scaled_f32",
  8,
  kAbiArgs_metal_navatala_sparse_csr_sp_m_v_scaled_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_residual_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_residual_f32 = {
  1,
  "navatala_sparse_csr_residual_f32",
  "metal",
  "navatala_sparse_csr_residual_f32",
  "kernel:metal:navatala_sparse_csr_residual_f32",
  "abi-r1:metal:navatala_sparse_csr_residual_f32",
  "abi-r1:metal:navatala_sparse_csr_residual_f32",
  7,
  kAbiArgs_metal_navatala_sparse_csr_residual_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_axpy_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_axpy_f32 = {
  1,
  "navatala_sparse_axpy_f32",
  "metal",
  "navatala_sparse_axpy_f32",
  "kernel:metal:navatala_sparse_axpy_f32",
  "abi-r1:metal:navatala_sparse_axpy_f32",
  "abi-r1:metal:navatala_sparse_axpy_f32",
  4,
  kAbiArgs_metal_navatala_sparse_axpy_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_xpay_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_xpay_f32 = {
  1,
  "navatala_sparse_xpay_f32",
  "metal",
  "navatala_sparse_xpay_f32",
  "kernel:metal:navatala_sparse_xpay_f32",
  "abi-r1:metal:navatala_sparse_xpay_f32",
  "abi-r1:metal:navatala_sparse_xpay_f32",
  4,
  kAbiArgs_metal_navatala_sparse_xpay_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_dot_partials_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_dot_partials_f32 = {
  1,
  "navatala_sparse_dot_partials_f32",
  "metal",
  "navatala_sparse_dot_partials_f32",
  "kernel:metal:navatala_sparse_dot_partials_f32",
  "abi-r1:metal:navatala_sparse_dot_partials_f32",
  "abi-r1:metal:navatala_sparse_dot_partials_f32",
  4,
  kAbiArgs_metal_navatala_sparse_dot_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_scal_f32[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_scal_f32 = {
  1,
  "navatala_sparse_scal_f32",
  "metal",
  "navatala_sparse_scal_f32",
  "kernel:metal:navatala_sparse_scal_f32",
  "abi-r1:metal:navatala_sparse_scal_f32",
  "abi-r1:metal:navatala_sparse_scal_f32",
  3,
  kAbiArgs_metal_navatala_sparse_scal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_nrm2_partials_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_nrm2_partials_f32 = {
  1,
  "navatala_sparse_nrm2_partials_f32",
  "metal",
  "navatala_sparse_nrm2_partials_f32",
  "kernel:metal:navatala_sparse_nrm2_partials_f32",
  "abi-r1:metal:navatala_sparse_nrm2_partials_f32",
  "abi-r1:metal:navatala_sparse_nrm2_partials_f32",
  3,
  kAbiArgs_metal_navatala_sparse_nrm2_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_zero_fill_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_zero_fill_f32 = {
  1,
  "navatala_sparse_zero_fill_f32",
  "metal",
  "navatala_sparse_zero_fill_f32",
  "kernel:metal:navatala_sparse_zero_fill_f32",
  "abi-r1:metal:navatala_sparse_zero_fill_f32",
  "abi-r1:metal:navatala_sparse_zero_fill_f32",
  2,
  kAbiArgs_metal_navatala_sparse_zero_fill_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_zero_fill_u32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_zero_fill_u32 = {
  1,
  "navatala_sparse_zero_fill_u32",
  "metal",
  "navatala_sparse_zero_fill_u32",
  "kernel:metal:navatala_sparse_zero_fill_u32",
  "abi-r1:metal:navatala_sparse_zero_fill_u32",
  "abi-r1:metal:navatala_sparse_zero_fill_u32",
  2,
  kAbiArgs_metal_navatala_sparse_zero_fill_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_zero_scalar_u32[] = {
  { "x", 0, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_zero_scalar_u32 = {
  1,
  "navatala_sparse_zero_scalar_u32",
  "metal",
  "navatala_sparse_zero_scalar_u32",
  "kernel:metal:navatala_sparse_zero_scalar_u32",
  "abi-r1:metal:navatala_sparse_zero_scalar_u32",
  "abi-r1:metal:navatala_sparse_zero_scalar_u32",
  1,
  kAbiArgs_metal_navatala_sparse_zero_scalar_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_fill_neg_one_i32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_fill_neg_one_i32 = {
  1,
  "navatala_sparse_fill_neg_one_i32",
  "metal",
  "navatala_sparse_fill_neg_one_i32",
  "kernel:metal:navatala_sparse_fill_neg_one_i32",
  "abi-r1:metal:navatala_sparse_fill_neg_one_i32",
  "abi-r1:metal:navatala_sparse_fill_neg_one_i32",
  2,
  kAbiArgs_metal_navatala_sparse_fill_neg_one_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_copy_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_copy_f32 = {
  1,
  "navatala_sparse_copy_f32",
  "metal",
  "navatala_sparse_copy_f32",
  "kernel:metal:navatala_sparse_copy_f32",
  "abi-r1:metal:navatala_sparse_copy_f32",
  "abi-r1:metal:navatala_sparse_copy_f32",
  3,
  kAbiArgs_metal_navatala_sparse_copy_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bsr_sp_m_v_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nBlockRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "blockDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bsr_sp_m_v_f32 = {
  1,
  "navatala_sparse_bsr_sp_m_v_f32",
  "metal",
  "navatala_sparse_bsr_sp_m_v_f32",
  "kernel:metal:navatala_sparse_bsr_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_bsr_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_bsr_sp_m_v_f32",
  7,
  kAbiArgs_metal_navatala_sparse_bsr_sp_m_v_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_transpose_count_f32[] = {
  { "colIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "TrowCount", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_transpose_count_f32 = {
  1,
  "navatala_sparse_csr_transpose_count_f32",
  "metal",
  "navatala_sparse_csr_transpose_count_f32",
  "kernel:metal:navatala_sparse_csr_transpose_count_f32",
  "abi-r1:metal:navatala_sparse_csr_transpose_count_f32",
  "abi-r1:metal:navatala_sparse_csr_transpose_count_f32",
  4,
  kAbiArgs_metal_navatala_sparse_csr_transpose_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_transpose_fill_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "TrowPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "TcolIdx", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Tvalues", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "TwritePos", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_transpose_fill_f32 = {
  1,
  "navatala_sparse_csr_transpose_fill_f32",
  "metal",
  "navatala_sparse_csr_transpose_fill_f32",
  "kernel:metal:navatala_sparse_csr_transpose_fill_f32",
  "abi-r1:metal:navatala_sparse_csr_transpose_fill_f32",
  "abi-r1:metal:navatala_sparse_csr_transpose_fill_f32",
  8,
  kAbiArgs_metal_navatala_sparse_csr_transpose_fill_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_row_norm_l1_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNorm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_row_norm_l1_f32 = {
  1,
  "navatala_sparse_csr_row_norm_l1_f32",
  "metal",
  "navatala_sparse_csr_row_norm_l1_f32",
  "kernel:metal:navatala_sparse_csr_row_norm_l1_f32",
  "abi-r1:metal:navatala_sparse_csr_row_norm_l1_f32",
  "abi-r1:metal:navatala_sparse_csr_row_norm_l1_f32",
  4,
  kAbiArgs_metal_navatala_sparse_csr_row_norm_l1_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_row_norm_linf_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNorm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_row_norm_linf_f32 = {
  1,
  "navatala_sparse_csr_row_norm_linf_f32",
  "metal",
  "navatala_sparse_csr_row_norm_linf_f32",
  "kernel:metal:navatala_sparse_csr_row_norm_linf_f32",
  "abi-r1:metal:navatala_sparse_csr_row_norm_linf_f32",
  "abi-r1:metal:navatala_sparse_csr_row_norm_linf_f32",
  4,
  kAbiArgs_metal_navatala_sparse_csr_row_norm_linf_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_frobenius_partials_f32[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_frobenius_partials_f32 = {
  1,
  "navatala_sparse_csr_frobenius_partials_f32",
  "metal",
  "navatala_sparse_csr_frobenius_partials_f32",
  "kernel:metal:navatala_sparse_csr_frobenius_partials_f32",
  "abi-r1:metal:navatala_sparse_csr_frobenius_partials_f32",
  "abi-r1:metal:navatala_sparse_csr_frobenius_partials_f32",
  3,
  kAbiArgs_metal_navatala_sparse_csr_frobenius_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_truncate_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tol", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newRowNnz", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_truncate_f32 = {
  1,
  "navatala_sparse_csr_truncate_f32",
  "metal",
  "navatala_sparse_csr_truncate_f32",
  "kernel:metal:navatala_sparse_csr_truncate_f32",
  "abi-r1:metal:navatala_sparse_csr_truncate_f32",
  "abi-r1:metal:navatala_sparse_csr_truncate_f32",
  6,
  kAbiArgs_metal_navatala_sparse_csr_truncate_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_compact_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newRowPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "keepMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newColIdx", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newValues", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_compact_f32 = {
  1,
  "navatala_sparse_csr_compact_f32",
  "metal",
  "navatala_sparse_csr_compact_f32",
  "kernel:metal:navatala_sparse_csr_compact_f32",
  "abi-r1:metal:navatala_sparse_csr_compact_f32",
  "abi-r1:metal:navatala_sparse_csr_compact_f32",
  8,
  kAbiArgs_metal_navatala_sparse_csr_compact_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_sort_columns_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_sort_columns_f32 = {
  1,
  "navatala_sparse_csr_sort_columns_f32",
  "metal",
  "navatala_sparse_csr_sort_columns_f32",
  "kernel:metal:navatala_sparse_csr_sort_columns_f32",
  "abi-r1:metal:navatala_sparse_csr_sort_columns_f32",
  "abi-r1:metal:navatala_sparse_csr_sort_columns_f32",
  4,
  kAbiArgs_metal_navatala_sparse_csr_sort_columns_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_permute_rows_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "perm", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newRowPtr", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newColIdx", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newValues", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_permute_rows_f32 = {
  1,
  "navatala_sparse_csr_permute_rows_f32",
  "metal",
  "navatala_sparse_csr_permute_rows_f32",
  "kernel:metal:navatala_sparse_csr_permute_rows_f32",
  "abi-r1:metal:navatala_sparse_csr_permute_rows_f32",
  "abi-r1:metal:navatala_sparse_csr_permute_rows_f32",
  8,
  kAbiArgs_metal_navatala_sparse_csr_permute_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_check_diag_dominance_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isDiagDom", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "allDiagDom", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_check_diag_dominance_f32 = {
  1,
  "navatala_sparse_csr_check_diag_dominance_f32",
  "metal",
  "navatala_sparse_csr_check_diag_dominance_f32",
  "kernel:metal:navatala_sparse_csr_check_diag_dominance_f32",
  "abi-r1:metal:navatala_sparse_csr_check_diag_dominance_f32",
  "abi-r1:metal:navatala_sparse_csr_check_diag_dominance_f32",
  6,
  kAbiArgs_metal_navatala_sparse_csr_check_diag_dominance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_row_nnz_histogram[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBins", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_row_nnz_histogram = {
  1,
  "navatala_sparse_csr_row_nnz_histogram",
  "metal",
  "navatala_sparse_csr_row_nnz_histogram",
  "kernel:metal:navatala_sparse_csr_row_nnz_histogram",
  "abi-r1:metal:navatala_sparse_csr_row_nnz_histogram",
  "abi-r1:metal:navatala_sparse_csr_row_nnz_histogram",
  4,
  kAbiArgs_metal_navatala_sparse_csr_row_nnz_histogram
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_dense_l_u_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pivots", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_dense_l_u_f32 = {
  1,
  "navatala_sparse_dense_l_u_f32",
  "metal",
  "navatala_sparse_dense_l_u_f32",
  "kernel:metal:navatala_sparse_dense_l_u_f32",
  "abi-r1:metal:navatala_sparse_dense_l_u_f32",
  "abi-r1:metal:navatala_sparse_dense_l_u_f32",
  3,
  kAbiArgs_metal_navatala_sparse_dense_l_u_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_dense_l_u_solve_f32[] = {
  { "LU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pivots", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_dense_l_u_solve_f32 = {
  1,
  "navatala_sparse_dense_l_u_solve_f32",
  "metal",
  "navatala_sparse_dense_l_u_solve_f32",
  "kernel:metal:navatala_sparse_dense_l_u_solve_f32",
  "abi-r1:metal:navatala_sparse_dense_l_u_solve_f32",
  "abi-r1:metal:navatala_sparse_dense_l_u_solve_f32",
  5,
  kAbiArgs_metal_navatala_sparse_dense_l_u_solve_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_replace_values_f32[] = {
  { "newValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_replace_values_f32 = {
  1,
  "navatala_sparse_csr_replace_values_f32",
  "metal",
  "navatala_sparse_csr_replace_values_f32",
  "kernel:metal:navatala_sparse_csr_replace_values_f32",
  "abi-r1:metal:navatala_sparse_csr_replace_values_f32",
  "abi-r1:metal:navatala_sparse_csr_replace_values_f32",
  3,
  kAbiArgs_metal_navatala_sparse_csr_replace_values_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_to_coo_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowIdx", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cooColIdx", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cooValues", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_to_coo_f32 = {
  1,
  "navatala_sparse_csr_to_coo_f32",
  "metal",
  "navatala_sparse_csr_to_coo_f32",
  "kernel:metal:navatala_sparse_csr_to_coo_f32",
  "abi-r1:metal:navatala_sparse_csr_to_coo_f32",
  "abi-r1:metal:navatala_sparse_csr_to_coo_f32",
  7,
  kAbiArgs_metal_navatala_sparse_csr_to_coo_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_coo_to_csr_row_count[] = {
  { "rowIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowCount", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_coo_to_csr_row_count = {
  1,
  "navatala_sparse_coo_to_csr_row_count",
  "metal",
  "navatala_sparse_coo_to_csr_row_count",
  "kernel:metal:navatala_sparse_coo_to_csr_row_count",
  "abi-r1:metal:navatala_sparse_coo_to_csr_row_count",
  "abi-r1:metal:navatala_sparse_coo_to_csr_row_count",
  4,
  kAbiArgs_metal_navatala_sparse_coo_to_csr_row_count
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_to_ell_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNnzPerRow", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ellColIdx", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ellValues", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_to_ell_f32 = {
  1,
  "navatala_sparse_csr_to_ell_f32",
  "metal",
  "navatala_sparse_csr_to_ell_f32",
  "kernel:metal:navatala_sparse_csr_to_ell_f32",
  "abi-r1:metal:navatala_sparse_csr_to_ell_f32",
  "abi-r1:metal:navatala_sparse_csr_to_ell_f32",
  7,
  kAbiArgs_metal_navatala_sparse_csr_to_ell_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ell_sp_m_v_f32[] = {
  { "ellColIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ellValues", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNnzPerRow", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ell_sp_m_v_f32 = {
  1,
  "navatala_sparse_ell_sp_m_v_f32",
  "metal",
  "navatala_sparse_ell_sp_m_v_f32",
  "kernel:metal:navatala_sparse_ell_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_ell_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_ell_sp_m_v_f32",
  6,
  kAbiArgs_metal_navatala_sparse_ell_sp_m_v_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_add_symbolic_row_nnz[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BrowPtr", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BcolIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "CrowNnz", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_add_symbolic_row_nnz = {
  1,
  "navatala_sparse_csr_add_symbolic_row_nnz",
  "metal",
  "navatala_sparse_csr_add_symbolic_row_nnz",
  "kernel:metal:navatala_sparse_csr_add_symbolic_row_nnz",
  "abi-r1:metal:navatala_sparse_csr_add_symbolic_row_nnz",
  "abi-r1:metal:navatala_sparse_csr_add_symbolic_row_nnz",
  6,
  kAbiArgs_metal_navatala_sparse_csr_add_symbolic_row_nnz
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_add_numeric_f32[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Avalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BrowPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BcolIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Bvalues", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "CrowPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "CcolIdx", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Cvalues", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_add_numeric_f32 = {
  1,
  "navatala_sparse_csr_add_numeric_f32",
  "metal",
  "navatala_sparse_csr_add_numeric_f32",
  "kernel:metal:navatala_sparse_csr_add_numeric_f32",
  "abi-r1:metal:navatala_sparse_csr_add_numeric_f32",
  "abi-r1:metal:navatala_sparse_csr_add_numeric_f32",
  12,
  kAbiArgs_metal_navatala_sparse_csr_add_numeric_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_scale_f32[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_scale_f32 = {
  1,
  "navatala_sparse_csr_scale_f32",
  "metal",
  "navatala_sparse_csr_scale_f32",
  "kernel:metal:navatala_sparse_csr_scale_f32",
  "abi-r1:metal:navatala_sparse_csr_scale_f32",
  "abi-r1:metal:navatala_sparse_csr_scale_f32",
  3,
  kAbiArgs_metal_navatala_sparse_csr_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_symmetry_check_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tol", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "asymmetricFlags", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "isSymmetric", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_symmetry_check_f32 = {
  1,
  "navatala_sparse_csr_symmetry_check_f32",
  "metal",
  "navatala_sparse_csr_symmetry_check_f32",
  "kernel:metal:navatala_sparse_csr_symmetry_check_f32",
  "abi-r1:metal:navatala_sparse_csr_symmetry_check_f32",
  "abi-r1:metal:navatala_sparse_csr_symmetry_check_f32",
  7,
  kAbiArgs_metal_navatala_sparse_csr_symmetry_check_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_nnz_per_row[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_nnz_per_row = {
  1,
  "navatala_sparse_csr_nnz_per_row",
  "metal",
  "navatala_sparse_csr_nnz_per_row",
  "kernel:metal:navatala_sparse_csr_nnz_per_row",
  "abi-r1:metal:navatala_sparse_csr_nnz_per_row",
  "abi-r1:metal:navatala_sparse_csr_nnz_per_row",
  3,
  kAbiArgs_metal_navatala_sparse_csr_nnz_per_row
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_max_nnz_per_row_partials[] = {
  { "nnzPerRow", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_max_nnz_per_row_partials = {
  1,
  "navatala_sparse_csr_max_nnz_per_row_partials",
  "metal",
  "navatala_sparse_csr_max_nnz_per_row_partials",
  "kernel:metal:navatala_sparse_csr_max_nnz_per_row_partials",
  "abi-r1:metal:navatala_sparse_csr_max_nnz_per_row_partials",
  "abi-r1:metal:navatala_sparse_csr_max_nnz_per_row_partials",
  3,
  kAbiArgs_metal_navatala_sparse_csr_max_nnz_per_row_partials
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_extract_diag_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "diag", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_extract_diag_f32 = {
  1,
  "navatala_sparse_csr_extract_diag_f32",
  "metal",
  "navatala_sparse_csr_extract_diag_f32",
  "kernel:metal:navatala_sparse_csr_extract_diag_f32",
  "abi-r1:metal:navatala_sparse_csr_extract_diag_f32",
  "abi-r1:metal:navatala_sparse_csr_extract_diag_f32",
  5,
  kAbiArgs_metal_navatala_sparse_csr_extract_diag_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_permute_cols_f32[] = {
  { "colIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "perm", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnz", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newColIdx", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newValues", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_permute_cols_f32 = {
  1,
  "navatala_sparse_csr_permute_cols_f32",
  "metal",
  "navatala_sparse_csr_permute_cols_f32",
  "kernel:metal:navatala_sparse_csr_permute_cols_f32",
  "abi-r1:metal:navatala_sparse_csr_permute_cols_f32",
  "abi-r1:metal:navatala_sparse_csr_permute_cols_f32",
  6,
  kAbiArgs_metal_navatala_sparse_csr_permute_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_row_slice_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowBegin", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "subRowPtr", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "subColIdx", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "subValues", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_row_slice_f32 = {
  1,
  "navatala_sparse_csr_row_slice_f32",
  "metal",
  "navatala_sparse_csr_row_slice_f32",
  "kernel:metal:navatala_sparse_csr_row_slice_f32",
  "abi-r1:metal:navatala_sparse_csr_row_slice_f32",
  "abi-r1:metal:navatala_sparse_csr_row_slice_f32",
  8,
  kAbiArgs_metal_navatala_sparse_csr_row_slice_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_update_x_r_f32[] = {
  { "p", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_update_x_r_f32 = {
  1,
  "navatala_sparse_cg_update_x_r_f32",
  "metal",
  "navatala_sparse_cg_update_x_r_f32",
  "kernel:metal:navatala_sparse_cg_update_x_r_f32",
  "abi-r1:metal:navatala_sparse_cg_update_x_r_f32",
  "abi-r1:metal:navatala_sparse_cg_update_x_r_f32",
  6,
  kAbiArgs_metal_navatala_sparse_cg_update_x_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_update_p_f32[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_update_p_f32 = {
  1,
  "navatala_sparse_cg_update_p_f32",
  "metal",
  "navatala_sparse_cg_update_p_f32",
  "kernel:metal:navatala_sparse_cg_update_p_f32",
  "abi-r1:metal:navatala_sparse_cg_update_p_f32",
  "abi-r1:metal:navatala_sparse_cg_update_p_f32",
  4,
  kAbiArgs_metal_navatala_sparse_cg_update_p_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_compute_alpha_f32[] = {
  { "rTz", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pTAp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_compute_alpha_f32 = {
  1,
  "navatala_sparse_cg_compute_alpha_f32",
  "metal",
  "navatala_sparse_cg_compute_alpha_f32",
  "kernel:metal:navatala_sparse_cg_compute_alpha_f32",
  "abi-r1:metal:navatala_sparse_cg_compute_alpha_f32",
  "abi-r1:metal:navatala_sparse_cg_compute_alpha_f32",
  3,
  kAbiArgs_metal_navatala_sparse_cg_compute_alpha_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_compute_beta_f32[] = {
  { "rTzNew", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rTzOld", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_compute_beta_f32 = {
  1,
  "navatala_sparse_cg_compute_beta_f32",
  "metal",
  "navatala_sparse_cg_compute_beta_f32",
  "kernel:metal:navatala_sparse_cg_compute_beta_f32",
  "abi-r1:metal:navatala_sparse_cg_compute_beta_f32",
  "abi-r1:metal:navatala_sparse_cg_compute_beta_f32",
  3,
  kAbiArgs_metal_navatala_sparse_cg_compute_beta_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_fused_update_dot_f32[] = {
  { "p", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rTzPartials", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_fused_update_dot_f32 = {
  1,
  "navatala_sparse_cg_fused_update_dot_f32",
  "metal",
  "navatala_sparse_cg_fused_update_dot_f32",
  "kernel:metal:navatala_sparse_cg_fused_update_dot_f32",
  "abi-r1:metal:navatala_sparse_cg_fused_update_dot_f32",
  "abi-r1:metal:navatala_sparse_cg_fused_update_dot_f32",
  8,
  kAbiArgs_metal_navatala_sparse_cg_fused_update_dot_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_init_residual_f32[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_init_residual_f32 = {
  1,
  "navatala_sparse_cg_init_residual_f32",
  "metal",
  "navatala_sparse_cg_init_residual_f32",
  "kernel:metal:navatala_sparse_cg_init_residual_f32",
  "abi-r1:metal:navatala_sparse_cg_init_residual_f32",
  "abi-r1:metal:navatala_sparse_cg_init_residual_f32",
  4,
  kAbiArgs_metal_navatala_sparse_cg_init_residual_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cg_residual_norm_partials_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cg_residual_norm_partials_f32 = {
  1,
  "navatala_sparse_cg_residual_norm_partials_f32",
  "metal",
  "navatala_sparse_cg_residual_norm_partials_f32",
  "kernel:metal:navatala_sparse_cg_residual_norm_partials_f32",
  "abi-r1:metal:navatala_sparse_cg_residual_norm_partials_f32",
  "abi-r1:metal:navatala_sparse_cg_residual_norm_partials_f32",
  3,
  kAbiArgs_metal_navatala_sparse_cg_residual_norm_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_update_p_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_update_p_f32 = {
  1,
  "navatala_sparse_bicgstab_update_p_f32",
  "metal",
  "navatala_sparse_bicgstab_update_p_f32",
  "kernel:metal:navatala_sparse_bicgstab_update_p_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_update_p_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_update_p_f32",
  6,
  kAbiArgs_metal_navatala_sparse_bicgstab_update_p_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_update_x_f32[] = {
  { "phat", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shat", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_update_x_f32 = {
  1,
  "navatala_sparse_bicgstab_update_x_f32",
  "metal",
  "navatala_sparse_bicgstab_update_x_f32",
  "kernel:metal:navatala_sparse_bicgstab_update_x_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_update_x_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_update_x_f32",
  6,
  kAbiArgs_metal_navatala_sparse_bicgstab_update_x_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_compute_beta_f32[] = {
  { "rhoNew", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rhoOld", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_compute_beta_f32 = {
  1,
  "navatala_sparse_bicgstab_compute_beta_f32",
  "metal",
  "navatala_sparse_bicgstab_compute_beta_f32",
  "kernel:metal:navatala_sparse_bicgstab_compute_beta_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_beta_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_beta_f32",
  5,
  kAbiArgs_metal_navatala_sparse_bicgstab_compute_beta_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_compute_omega_f32[] = {
  { "dotTS", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotTT", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_compute_omega_f32 = {
  1,
  "navatala_sparse_bicgstab_compute_omega_f32",
  "metal",
  "navatala_sparse_bicgstab_compute_omega_f32",
  "kernel:metal:navatala_sparse_bicgstab_compute_omega_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_omega_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_omega_f32",
  3,
  kAbiArgs_metal_navatala_sparse_bicgstab_compute_omega_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_update_r_f32[] = {
  { "s", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "t", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_update_r_f32 = {
  1,
  "navatala_sparse_bicgstab_update_r_f32",
  "metal",
  "navatala_sparse_bicgstab_update_r_f32",
  "kernel:metal:navatala_sparse_bicgstab_update_r_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_update_r_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_update_r_f32",
  5,
  kAbiArgs_metal_navatala_sparse_bicgstab_update_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_compute_s_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "s", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_compute_s_f32 = {
  1,
  "navatala_sparse_bicgstab_compute_s_f32",
  "metal",
  "navatala_sparse_bicgstab_compute_s_f32",
  "kernel:metal:navatala_sparse_bicgstab_compute_s_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_s_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_s_f32",
  5,
  kAbiArgs_metal_navatala_sparse_bicgstab_compute_s_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_compute_alpha_f32[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r0hatV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_compute_alpha_f32 = {
  1,
  "navatala_sparse_bicgstab_compute_alpha_f32",
  "metal",
  "navatala_sparse_bicgstab_compute_alpha_f32",
  "kernel:metal:navatala_sparse_bicgstab_compute_alpha_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_alpha_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_compute_alpha_f32",
  3,
  kAbiArgs_metal_navatala_sparse_bicgstab_compute_alpha_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bicgstab_fused_update_all_f32[] = {
  { "phat", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shat", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "t", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bicgstab_fused_update_all_f32 = {
  1,
  "navatala_sparse_bicgstab_fused_update_all_f32",
  "metal",
  "navatala_sparse_bicgstab_fused_update_all_f32",
  "kernel:metal:navatala_sparse_bicgstab_fused_update_all_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_fused_update_all_f32",
  "abi-r1:metal:navatala_sparse_bicgstab_fused_update_all_f32",
  9,
  kAbiArgs_metal_navatala_sparse_bicgstab_fused_update_all_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_arnoldi_project_f32[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "w", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_arnoldi_project_f32 = {
  1,
  "navatala_sparse_gmres_arnoldi_project_f32",
  "metal",
  "navatala_sparse_gmres_arnoldi_project_f32",
  "kernel:metal:navatala_sparse_gmres_arnoldi_project_f32",
  "abi-r1:metal:navatala_sparse_gmres_arnoldi_project_f32",
  "abi-r1:metal:navatala_sparse_gmres_arnoldi_project_f32",
  4,
  kAbiArgs_metal_navatala_sparse_gmres_arnoldi_project_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_normalize_f32[] = {
  { "w", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "norm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_normalize_f32 = {
  1,
  "navatala_sparse_gmres_normalize_f32",
  "metal",
  "navatala_sparse_gmres_normalize_f32",
  "kernel:metal:navatala_sparse_gmres_normalize_f32",
  "abi-r1:metal:navatala_sparse_gmres_normalize_f32",
  "abi-r1:metal:navatala_sparse_gmres_normalize_f32",
  4,
  kAbiArgs_metal_navatala_sparse_gmres_normalize_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_update_solution_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_update_solution_f32 = {
  1,
  "navatala_sparse_gmres_update_solution_f32",
  "metal",
  "navatala_sparse_gmres_update_solution_f32",
  "kernel:metal:navatala_sparse_gmres_update_solution_f32",
  "abi-r1:metal:navatala_sparse_gmres_update_solution_f32",
  "abi-r1:metal:navatala_sparse_gmres_update_solution_f32",
  5,
  kAbiArgs_metal_navatala_sparse_gmres_update_solution_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_givens_rotation_f32[] = {
  { "cs", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "h", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csNew", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "snNew", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_givens_rotation_f32 = {
  1,
  "navatala_sparse_gmres_givens_rotation_f32",
  "metal",
  "navatala_sparse_gmres_givens_rotation_f32",
  "kernel:metal:navatala_sparse_gmres_givens_rotation_f32",
  "abi-r1:metal:navatala_sparse_gmres_givens_rotation_f32",
  "abi-r1:metal:navatala_sparse_gmres_givens_rotation_f32",
  6,
  kAbiArgs_metal_navatala_sparse_gmres_givens_rotation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_arnoldi_batch_project_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hCol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "j", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "w", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_arnoldi_batch_project_f32 = {
  1,
  "navatala_sparse_gmres_arnoldi_batch_project_f32",
  "metal",
  "navatala_sparse_gmres_arnoldi_batch_project_f32",
  "kernel:metal:navatala_sparse_gmres_arnoldi_batch_project_f32",
  "abi-r1:metal:navatala_sparse_gmres_arnoldi_batch_project_f32",
  "abi-r1:metal:navatala_sparse_gmres_arnoldi_batch_project_f32",
  5,
  kAbiArgs_metal_navatala_sparse_gmres_arnoldi_batch_project_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_hessenberg_dot_partials_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "j", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_hessenberg_dot_partials_f32 = {
  1,
  "navatala_sparse_gmres_hessenberg_dot_partials_f32",
  "metal",
  "navatala_sparse_gmres_hessenberg_dot_partials_f32",
  "kernel:metal:navatala_sparse_gmres_hessenberg_dot_partials_f32",
  "abi-r1:metal:navatala_sparse_gmres_hessenberg_dot_partials_f32",
  "abi-r1:metal:navatala_sparse_gmres_hessenberg_dot_partials_f32",
  5,
  kAbiArgs_metal_navatala_sparse_gmres_hessenberg_dot_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_restart_init_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_restart_init_f32 = {
  1,
  "navatala_sparse_gmres_restart_init_f32",
  "metal",
  "navatala_sparse_gmres_restart_init_f32",
  "kernel:metal:navatala_sparse_gmres_restart_init_f32",
  "abi-r1:metal:navatala_sparse_gmres_restart_init_f32",
  "abi-r1:metal:navatala_sparse_gmres_restart_init_f32",
  4,
  kAbiArgs_metal_navatala_sparse_gmres_restart_init_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_back_solve_f32[] = {
  { "H", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "g", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_back_solve_f32 = {
  1,
  "navatala_sparse_gmres_back_solve_f32",
  "metal",
  "navatala_sparse_gmres_back_solve_f32",
  "kernel:metal:navatala_sparse_gmres_back_solve_f32",
  "abi-r1:metal:navatala_sparse_gmres_back_solve_f32",
  "abi-r1:metal:navatala_sparse_gmres_back_solve_f32",
  4,
  kAbiArgs_metal_navatala_sparse_gmres_back_solve_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_gmres_store_basis_f32[] = {
  { "w", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_gmres_store_basis_f32 = {
  1,
  "navatala_sparse_gmres_store_basis_f32",
  "metal",
  "navatala_sparse_gmres_store_basis_f32",
  "kernel:metal:navatala_sparse_gmres_store_basis_f32",
  "abi-r1:metal:navatala_sparse_gmres_store_basis_f32",
  "abi-r1:metal:navatala_sparse_gmres_store_basis_f32",
  4,
  kAbiArgs_metal_navatala_sparse_gmres_store_basis_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_shadow_project_f32[] = {
  { "P", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "s", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_shadow_project_f32 = {
  1,
  "navatala_sparse_idr_shadow_project_f32",
  "metal",
  "navatala_sparse_idr_shadow_project_f32",
  "kernel:metal:navatala_sparse_idr_shadow_project_f32",
  "abi-r1:metal:navatala_sparse_idr_shadow_project_f32",
  "abi-r1:metal:navatala_sparse_idr_shadow_project_f32",
  5,
  kAbiArgs_metal_navatala_sparse_idr_shadow_project_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_update_x_r_f32[] = {
  { "t", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_update_x_r_f32 = {
  1,
  "navatala_sparse_idr_update_x_r_f32",
  "metal",
  "navatala_sparse_idr_update_x_r_f32",
  "kernel:metal:navatala_sparse_idr_update_x_r_f32",
  "abi-r1:metal:navatala_sparse_idr_update_x_r_f32",
  "abi-r1:metal:navatala_sparse_idr_update_x_r_f32",
  6,
  kAbiArgs_metal_navatala_sparse_idr_update_x_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_g_space_update_f32[] = {
  { "dRj", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dRk", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_g_space_update_f32 = {
  1,
  "navatala_sparse_idr_g_space_update_f32",
  "metal",
  "navatala_sparse_idr_g_space_update_f32",
  "kernel:metal:navatala_sparse_idr_g_space_update_f32",
  "abi-r1:metal:navatala_sparse_idr_g_space_update_f32",
  "abi-r1:metal:navatala_sparse_idr_g_space_update_f32",
  4,
  kAbiArgs_metal_navatala_sparse_idr_g_space_update_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_compute_omega_f32[] = {
  { "dotTS", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotTT", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_compute_omega_f32 = {
  1,
  "navatala_sparse_idr_compute_omega_f32",
  "metal",
  "navatala_sparse_idr_compute_omega_f32",
  "kernel:metal:navatala_sparse_idr_compute_omega_f32",
  "abi-r1:metal:navatala_sparse_idr_compute_omega_f32",
  "abi-r1:metal:navatala_sparse_idr_compute_omega_f32",
  3,
  kAbiArgs_metal_navatala_sparse_idr_compute_omega_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_build_shadow_space_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "s", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "P", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_build_shadow_space_f32 = {
  1,
  "navatala_sparse_idr_build_shadow_space_f32",
  "metal",
  "navatala_sparse_idr_build_shadow_space_f32",
  "kernel:metal:navatala_sparse_idr_build_shadow_space_f32",
  "abi-r1:metal:navatala_sparse_idr_build_shadow_space_f32",
  "abi-r1:metal:navatala_sparse_idr_build_shadow_space_f32",
  4,
  kAbiArgs_metal_navatala_sparse_idr_build_shadow_space_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_m_sync_fused_f32[] = {
  { "P", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Minv", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "s", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_m_sync_fused_f32 = {
  1,
  "navatala_sparse_idr_m_sync_fused_f32",
  "metal",
  "navatala_sparse_idr_m_sync_fused_f32",
  "kernel:metal:navatala_sparse_idr_m_sync_fused_f32",
  "abi-r1:metal:navatala_sparse_idr_m_sync_fused_f32",
  "abi-r1:metal:navatala_sparse_idr_m_sync_fused_f32",
  6,
  kAbiArgs_metal_navatala_sparse_idr_m_sync_fused_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_compute_v_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dR", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "c", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "s", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_compute_v_f32 = {
  1,
  "navatala_sparse_idr_compute_v_f32",
  "metal",
  "navatala_sparse_idr_compute_v_f32",
  "kernel:metal:navatala_sparse_idr_compute_v_f32",
  "abi-r1:metal:navatala_sparse_idr_compute_v_f32",
  "abi-r1:metal:navatala_sparse_idr_compute_v_f32",
  6,
  kAbiArgs_metal_navatala_sparse_idr_compute_v_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_idr_store_d_r_column_f32[] = {
  { "rNew", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rOld", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dR", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_idr_store_d_r_column_f32 = {
  1,
  "navatala_sparse_idr_store_d_r_column_f32",
  "metal",
  "navatala_sparse_idr_store_d_r_column_f32",
  "kernel:metal:navatala_sparse_idr_store_d_r_column_f32",
  "abi-r1:metal:navatala_sparse_idr_store_d_r_column_f32",
  "abi-r1:metal:navatala_sparse_idr_store_d_r_column_f32",
  5,
  kAbiArgs_metal_navatala_sparse_idr_store_d_r_column_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_pcg_update_x_r_f32[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_pcg_update_x_r_f32 = {
  1,
  "navatala_sparse_pcg_update_x_r_f32",
  "metal",
  "navatala_sparse_pcg_update_x_r_f32",
  "kernel:metal:navatala_sparse_pcg_update_x_r_f32",
  "abi-r1:metal:navatala_sparse_pcg_update_x_r_f32",
  "abi-r1:metal:navatala_sparse_pcg_update_x_r_f32",
  6,
  kAbiArgs_metal_navatala_sparse_pcg_update_x_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_pcgf_update_p_f32[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "zPrev", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_pcgf_update_p_f32 = {
  1,
  "navatala_sparse_pcgf_update_p_f32",
  "metal",
  "navatala_sparse_pcgf_update_p_f32",
  "kernel:metal:navatala_sparse_pcgf_update_p_f32",
  "abi-r1:metal:navatala_sparse_pcgf_update_p_f32",
  "abi-r1:metal:navatala_sparse_pcgf_update_p_f32",
  6,
  kAbiArgs_metal_navatala_sparse_pcgf_update_p_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_pbicgstab_fused_s_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "s", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_pbicgstab_fused_s_f32 = {
  1,
  "navatala_sparse_pbicgstab_fused_s_f32",
  "metal",
  "navatala_sparse_pbicgstab_fused_s_f32",
  "kernel:metal:navatala_sparse_pbicgstab_fused_s_f32",
  "abi-r1:metal:navatala_sparse_pbicgstab_fused_s_f32",
  "abi-r1:metal:navatala_sparse_pbicgstab_fused_s_f32",
  5,
  kAbiArgs_metal_navatala_sparse_pbicgstab_fused_s_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_pbicgstab_triple_dot_partials_f32[] = {
  { "r0hat", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_pbicgstab_triple_dot_partials_f32 = {
  1,
  "navatala_sparse_pbicgstab_triple_dot_partials_f32",
  "metal",
  "navatala_sparse_pbicgstab_triple_dot_partials_f32",
  "kernel:metal:navatala_sparse_pbicgstab_triple_dot_partials_f32",
  "abi-r1:metal:navatala_sparse_pbicgstab_triple_dot_partials_f32",
  "abi-r1:metal:navatala_sparse_pbicgstab_triple_dot_partials_f32",
  6,
  kAbiArgs_metal_navatala_sparse_pbicgstab_triple_dot_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_fgmres_store_z_f32[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Z", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_fgmres_store_z_f32 = {
  1,
  "navatala_sparse_fgmres_store_z_f32",
  "metal",
  "navatala_sparse_fgmres_store_z_f32",
  "kernel:metal:navatala_sparse_fgmres_store_z_f32",
  "abi-r1:metal:navatala_sparse_fgmres_store_z_f32",
  "abi-r1:metal:navatala_sparse_fgmres_store_z_f32",
  4,
  kAbiArgs_metal_navatala_sparse_fgmres_store_z_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_fgmres_update_solution_f32[] = {
  { "Z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_fgmres_update_solution_f32 = {
  1,
  "navatala_sparse_fgmres_update_solution_f32",
  "metal",
  "navatala_sparse_fgmres_update_solution_f32",
  "kernel:metal:navatala_sparse_fgmres_update_solution_f32",
  "abi-r1:metal:navatala_sparse_fgmres_update_solution_f32",
  "abi-r1:metal:navatala_sparse_fgmres_update_solution_f32",
  5,
  kAbiArgs_metal_navatala_sparse_fgmres_update_solution_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_pipe_c_g_fused_f32[] = {
  { "p", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "partials", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_pipe_c_g_fused_f32 = {
  1,
  "navatala_sparse_pipe_c_g_fused_f32",
  "metal",
  "navatala_sparse_pipe_c_g_fused_f32",
  "kernel:metal:navatala_sparse_pipe_c_g_fused_f32",
  "abi-r1:metal:navatala_sparse_pipe_c_g_fused_f32",
  "abi-r1:metal:navatala_sparse_pipe_c_g_fused_f32",
  9,
  kAbiArgs_metal_navatala_sparse_pipe_c_g_fused_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_chronopoulos_c_g_scalars_f32[] = {
  { "gamma", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "delta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gammaOld", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_chronopoulos_c_g_scalars_f32 = {
  1,
  "navatala_sparse_chronopoulos_c_g_scalars_f32",
  "metal",
  "navatala_sparse_chronopoulos_c_g_scalars_f32",
  "kernel:metal:navatala_sparse_chronopoulos_c_g_scalars_f32",
  "abi-r1:metal:navatala_sparse_chronopoulos_c_g_scalars_f32",
  "abi-r1:metal:navatala_sparse_chronopoulos_c_g_scalars_f32",
  5,
  kAbiArgs_metal_navatala_sparse_chronopoulos_c_g_scalars_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_fgmres_m_g_s_project_f32[] = {
  { "vi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "w", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hij", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_fgmres_m_g_s_project_f32 = {
  1,
  "navatala_sparse_fgmres_m_g_s_project_f32",
  "metal",
  "navatala_sparse_fgmres_m_g_s_project_f32",
  "kernel:metal:navatala_sparse_fgmres_m_g_s_project_f32",
  "abi-r1:metal:navatala_sparse_fgmres_m_g_s_project_f32",
  "abi-r1:metal:navatala_sparse_fgmres_m_g_s_project_f32",
  4,
  kAbiArgs_metal_navatala_sparse_fgmres_m_g_s_project_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_pbicgstab_fused_update_conv_f32[] = {
  { "phat", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shat", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "t", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omega", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rrPartials", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_pbicgstab_fused_update_conv_f32 = {
  1,
  "navatala_sparse_pbicgstab_fused_update_conv_f32",
  "metal",
  "navatala_sparse_pbicgstab_fused_update_conv_f32",
  "kernel:metal:navatala_sparse_pbicgstab_fused_update_conv_f32",
  "abi-r1:metal:navatala_sparse_pbicgstab_fused_update_conv_f32",
  "abi-r1:metal:navatala_sparse_pbicgstab_fused_update_conv_f32",
  10,
  kAbiArgs_metal_navatala_sparse_pbicgstab_fused_update_conv_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_jacobi_sweep_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_jacobi_sweep_f32 = {
  1,
  "navatala_sparse_jacobi_sweep_f32",
  "metal",
  "navatala_sparse_jacobi_sweep_f32",
  "kernel:metal:navatala_sparse_jacobi_sweep_f32",
  "abi-r1:metal:navatala_sparse_jacobi_sweep_f32",
  "abi-r1:metal:navatala_sparse_jacobi_sweep_f32",
  9,
  kAbiArgs_metal_navatala_sparse_jacobi_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_jacobi_l1_sweep_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_jacobi_l1_sweep_f32 = {
  1,
  "navatala_sparse_jacobi_l1_sweep_f32",
  "metal",
  "navatala_sparse_jacobi_l1_sweep_f32",
  "kernel:metal:navatala_sparse_jacobi_l1_sweep_f32",
  "abi-r1:metal:navatala_sparse_jacobi_l1_sweep_f32",
  "abi-r1:metal:navatala_sparse_jacobi_l1_sweep_f32",
  8,
  kAbiArgs_metal_navatala_sparse_jacobi_l1_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_extract_diagonal_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "diag", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_extract_diagonal_f32 = {
  1,
  "navatala_sparse_extract_diagonal_f32",
  "metal",
  "navatala_sparse_extract_diagonal_f32",
  "kernel:metal:navatala_sparse_extract_diagonal_f32",
  "abi-r1:metal:navatala_sparse_extract_diagonal_f32",
  "abi-r1:metal:navatala_sparse_extract_diagonal_f32",
  5,
  kAbiArgs_metal_navatala_sparse_extract_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_block_jacobi_sweep_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagBlocks", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nBlockRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "blockDim", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_block_jacobi_sweep_f32 = {
  1,
  "navatala_sparse_block_jacobi_sweep_f32",
  "metal",
  "navatala_sparse_block_jacobi_sweep_f32",
  "kernel:metal:navatala_sparse_block_jacobi_sweep_f32",
  "abi-r1:metal:navatala_sparse_block_jacobi_sweep_f32",
  "abi-r1:metal:navatala_sparse_block_jacobi_sweep_f32",
  10,
  kAbiArgs_metal_navatala_sparse_block_jacobi_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_jacobi_sweep_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaC", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "omegaF", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_jacobi_sweep_f32 = {
  1,
  "navatala_sparse_cf_jacobi_sweep_f32",
  "metal",
  "navatala_sparse_cf_jacobi_sweep_f32",
  "kernel:metal:navatala_sparse_cf_jacobi_sweep_f32",
  "abi-r1:metal:navatala_sparse_cf_jacobi_sweep_f32",
  "abi-r1:metal:navatala_sparse_cf_jacobi_sweep_f32",
  11,
  kAbiArgs_metal_navatala_sparse_cf_jacobi_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multicolor_gs_forward_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multicolor_gs_forward_f32 = {
  1,
  "navatala_sparse_multicolor_gs_forward_f32",
  "metal",
  "navatala_sparse_multicolor_gs_forward_f32",
  "kernel:metal:navatala_sparse_multicolor_gs_forward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_gs_forward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_gs_forward_f32",
  8,
  kAbiArgs_metal_navatala_sparse_multicolor_gs_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multicolor_gs_backward_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multicolor_gs_backward_f32 = {
  1,
  "navatala_sparse_multicolor_gs_backward_f32",
  "metal",
  "navatala_sparse_multicolor_gs_backward_f32",
  "kernel:metal:navatala_sparse_multicolor_gs_backward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_gs_backward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_gs_backward_f32",
  8,
  kAbiArgs_metal_navatala_sparse_multicolor_gs_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_fixcolor_gs_forward_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reorderPerm", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_fixcolor_gs_forward_f32 = {
  1,
  "navatala_sparse_fixcolor_gs_forward_f32",
  "metal",
  "navatala_sparse_fixcolor_gs_forward_f32",
  "kernel:metal:navatala_sparse_fixcolor_gs_forward_f32",
  "abi-r1:metal:navatala_sparse_fixcolor_gs_forward_f32",
  "abi-r1:metal:navatala_sparse_fixcolor_gs_forward_f32",
  9,
  kAbiArgs_metal_navatala_sparse_fixcolor_gs_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_fixcolor_gs_backward_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reorderPerm", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_fixcolor_gs_backward_f32 = {
  1,
  "navatala_sparse_fixcolor_gs_backward_f32",
  "metal",
  "navatala_sparse_fixcolor_gs_backward_f32",
  "kernel:metal:navatala_sparse_fixcolor_gs_backward_f32",
  "abi-r1:metal:navatala_sparse_fixcolor_gs_backward_f32",
  "abi-r1:metal:navatala_sparse_fixcolor_gs_backward_f32",
  9,
  kAbiArgs_metal_navatala_sparse_fixcolor_gs_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ilu_factor_color_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "values", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ilu_factor_color_f32 = {
  1,
  "navatala_sparse_ilu_factor_color_f32",
  "metal",
  "navatala_sparse_ilu_factor_color_f32",
  "kernel:metal:navatala_sparse_ilu_factor_color_f32",
  "abi-r1:metal:navatala_sparse_ilu_factor_color_f32",
  "abi-r1:metal:navatala_sparse_ilu_factor_color_f32",
  6,
  kAbiArgs_metal_navatala_sparse_ilu_factor_color_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ilu_forward_sweep_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "luValues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ilu_forward_sweep_f32 = {
  1,
  "navatala_sparse_ilu_forward_sweep_f32",
  "metal",
  "navatala_sparse_ilu_forward_sweep_f32",
  "kernel:metal:navatala_sparse_ilu_forward_sweep_f32",
  "abi-r1:metal:navatala_sparse_ilu_forward_sweep_f32",
  "abi-r1:metal:navatala_sparse_ilu_forward_sweep_f32",
  7,
  kAbiArgs_metal_navatala_sparse_ilu_forward_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ilu_backward_sweep_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "luValues", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ilu_backward_sweep_f32 = {
  1,
  "navatala_sparse_ilu_backward_sweep_f32",
  "metal",
  "navatala_sparse_ilu_backward_sweep_f32",
  "kernel:metal:navatala_sparse_ilu_backward_sweep_f32",
  "abi-r1:metal:navatala_sparse_ilu_backward_sweep_f32",
  "abi-r1:metal:navatala_sparse_ilu_backward_sweep_f32",
  8,
  kAbiArgs_metal_navatala_sparse_ilu_backward_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_dilu_build_reciprocal_d_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recipD", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_dilu_build_reciprocal_d_f32 = {
  1,
  "navatala_sparse_dilu_build_reciprocal_d_f32",
  "metal",
  "navatala_sparse_dilu_build_reciprocal_d_f32",
  "kernel:metal:navatala_sparse_dilu_build_reciprocal_d_f32",
  "abi-r1:metal:navatala_sparse_dilu_build_reciprocal_d_f32",
  "abi-r1:metal:navatala_sparse_dilu_build_reciprocal_d_f32",
  6,
  kAbiArgs_metal_navatala_sparse_dilu_build_reciprocal_d_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_dilu_apply_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recipD", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_dilu_apply_f32 = {
  1,
  "navatala_sparse_dilu_apply_f32",
  "metal",
  "navatala_sparse_dilu_apply_f32",
  "kernel:metal:navatala_sparse_dilu_apply_f32",
  "abi-r1:metal:navatala_sparse_dilu_apply_f32",
  "abi-r1:metal:navatala_sparse_dilu_apply_f32",
  9,
  kAbiArgs_metal_navatala_sparse_dilu_apply_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multicolor_dilu_forward_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recipD", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multicolor_dilu_forward_f32 = {
  1,
  "navatala_sparse_multicolor_dilu_forward_f32",
  "metal",
  "navatala_sparse_multicolor_dilu_forward_f32",
  "kernel:metal:navatala_sparse_multicolor_dilu_forward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_dilu_forward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_dilu_forward_f32",
  8,
  kAbiArgs_metal_navatala_sparse_multicolor_dilu_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multicolor_dilu_backward_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recipD", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOffsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsThisColor", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multicolor_dilu_backward_f32 = {
  1,
  "navatala_sparse_multicolor_dilu_backward_f32",
  "metal",
  "navatala_sparse_multicolor_dilu_backward_f32",
  "kernel:metal:navatala_sparse_multicolor_dilu_backward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_dilu_backward_f32",
  "abi-r1:metal:navatala_sparse_multicolor_dilu_backward_f32",
  8,
  kAbiArgs_metal_navatala_sparse_multicolor_dilu_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_chebyshev_sweep_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "d", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeffB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dNew", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_chebyshev_sweep_f32 = {
  1,
  "navatala_sparse_chebyshev_sweep_f32",
  "metal",
  "navatala_sparse_chebyshev_sweep_f32",
  "kernel:metal:navatala_sparse_chebyshev_sweep_f32",
  "abi-r1:metal:navatala_sparse_chebyshev_sweep_f32",
  "abi-r1:metal:navatala_sparse_chebyshev_sweep_f32",
  7,
  kAbiArgs_metal_navatala_sparse_chebyshev_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_chebyshev_coeff_update_f32[] = {
  { "theta", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "delta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sigma1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sigmaOld", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeffA", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeffB", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sigmaNew", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_chebyshev_coeff_update_f32 = {
  1,
  "navatala_sparse_chebyshev_coeff_update_f32",
  "metal",
  "navatala_sparse_chebyshev_coeff_update_f32",
  "kernel:metal:navatala_sparse_chebyshev_coeff_update_f32",
  "abi-r1:metal:navatala_sparse_chebyshev_coeff_update_f32",
  "abi-r1:metal:navatala_sparse_chebyshev_coeff_update_f32",
  7,
  kAbiArgs_metal_navatala_sparse_chebyshev_coeff_update_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_polynomial_sweep_f32[] = {
  { "Ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffs", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_polynomial_sweep_f32 = {
  1,
  "navatala_sparse_polynomial_sweep_f32",
  "metal",
  "navatala_sparse_polynomial_sweep_f32",
  "kernel:metal:navatala_sparse_polynomial_sweep_f32",
  "abi-r1:metal:navatala_sparse_polynomial_sweep_f32",
  "abi-r1:metal:navatala_sparse_polynomial_sweep_f32",
  7,
  kAbiArgs_metal_navatala_sparse_polynomial_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_kpz_polynomial_sweep_f32[] = {
  { "Ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kpzCoeffs", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xNew", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_kpz_polynomial_sweep_f32 = {
  1,
  "navatala_sparse_kpz_polynomial_sweep_f32",
  "metal",
  "navatala_sparse_kpz_polynomial_sweep_f32",
  "kernel:metal:navatala_sparse_kpz_polynomial_sweep_f32",
  "abi-r1:metal:navatala_sparse_kpz_polynomial_sweep_f32",
  "abi-r1:metal:navatala_sparse_kpz_polynomial_sweep_f32",
  7,
  kAbiArgs_metal_navatala_sparse_kpz_polynomial_sweep_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_kaczmarz_row_norm_sq_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNormSq", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_kaczmarz_row_norm_sq_f32 = {
  1,
  "navatala_sparse_kaczmarz_row_norm_sq_f32",
  "metal",
  "navatala_sparse_kaczmarz_row_norm_sq_f32",
  "kernel:metal:navatala_sparse_kaczmarz_row_norm_sq_f32",
  "abi-r1:metal:navatala_sparse_kaczmarz_row_norm_sq_f32",
  "abi-r1:metal:navatala_sparse_kaczmarz_row_norm_sq_f32",
  4,
  kAbiArgs_metal_navatala_sparse_kaczmarz_row_norm_sq_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_strength_of_connection_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "strongMask", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_strength_of_connection_f32 = {
  1,
  "navatala_sparse_strength_of_connection_f32",
  "metal",
  "navatala_sparse_strength_of_connection_f32",
  "kernel:metal:navatala_sparse_strength_of_connection_f32",
  "abi-r1:metal:navatala_sparse_strength_of_connection_f32",
  "abi-r1:metal:navatala_sparse_strength_of_connection_f32",
  6,
  kAbiArgs_metal_navatala_sparse_strength_of_connection_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_size2_propose_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pickArray", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_size2_propose_f32 = {
  1,
  "navatala_sparse_aggregate_size2_propose_f32",
  "metal",
  "navatala_sparse_aggregate_size2_propose_f32",
  "kernel:metal:navatala_sparse_aggregate_size2_propose_f32",
  "abi-r1:metal:navatala_sparse_aggregate_size2_propose_f32",
  "abi-r1:metal:navatala_sparse_aggregate_size2_propose_f32",
  6,
  kAbiArgs_metal_navatala_sparse_aggregate_size2_propose_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_size2_confirm[] = {
  { "pickArray", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_size2_confirm = {
  1,
  "navatala_sparse_aggregate_size2_confirm",
  "metal",
  "navatala_sparse_aggregate_size2_confirm",
  "kernel:metal:navatala_sparse_aggregate_size2_confirm",
  "abi-r1:metal:navatala_sparse_aggregate_size2_confirm",
  "abi-r1:metal:navatala_sparse_aggregate_size2_confirm",
  4,
  kAbiArgs_metal_navatala_sparse_aggregate_size2_confirm
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_promote_unmatched_singletons[] = {
  { "nRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_promote_unmatched_singletons = {
  1,
  "navatala_sparse_promote_unmatched_singletons",
  "metal",
  "navatala_sparse_promote_unmatched_singletons",
  "kernel:metal:navatala_sparse_promote_unmatched_singletons",
  "abi-r1:metal:navatala_sparse_promote_unmatched_singletons",
  "abi-r1:metal:navatala_sparse_promote_unmatched_singletons",
  2,
  kAbiArgs_metal_navatala_sparse_promote_unmatched_singletons
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_size4_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_size4_f32 = {
  1,
  "navatala_sparse_aggregate_size4_f32",
  "metal",
  "navatala_sparse_aggregate_size4_f32",
  "kernel:metal:navatala_sparse_aggregate_size4_f32",
  "abi-r1:metal:navatala_sparse_aggregate_size4_f32",
  "abi-r1:metal:navatala_sparse_aggregate_size4_f32",
  7,
  kAbiArgs_metal_navatala_sparse_aggregate_size4_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_size8_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_size8_f32 = {
  1,
  "navatala_sparse_aggregate_size8_f32",
  "metal",
  "navatala_sparse_aggregate_size8_f32",
  "kernel:metal:navatala_sparse_aggregate_size8_f32",
  "abi-r1:metal:navatala_sparse_aggregate_size8_f32",
  "abi-r1:metal:navatala_sparse_aggregate_size8_f32",
  7,
  kAbiArgs_metal_navatala_sparse_aggregate_size8_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_parallel_greedy_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_parallel_greedy_f32 = {
  1,
  "navatala_sparse_aggregate_parallel_greedy_f32",
  "metal",
  "navatala_sparse_aggregate_parallel_greedy_f32",
  "kernel:metal:navatala_sparse_aggregate_parallel_greedy_f32",
  "abi-r1:metal:navatala_sparse_aggregate_parallel_greedy_f32",
  "abi-r1:metal:navatala_sparse_aggregate_parallel_greedy_f32",
  8,
  kAbiArgs_metal_navatala_sparse_aggregate_parallel_greedy_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_assign_leader_compact_ids[] = {
  { "nRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "compactMap", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_assign_leader_compact_ids = {
  1,
  "navatala_sparse_assign_leader_compact_ids",
  "metal",
  "navatala_sparse_assign_leader_compact_ids",
  "kernel:metal:navatala_sparse_assign_leader_compact_ids",
  "abi-r1:metal:navatala_sparse_assign_leader_compact_ids",
  "abi-r1:metal:navatala_sparse_assign_leader_compact_ids",
  4,
  kAbiArgs_metal_navatala_sparse_assign_leader_compact_ids
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_apply_compact_ids[] = {
  { "compactMap", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_apply_compact_ids = {
  1,
  "navatala_sparse_apply_compact_ids",
  "metal",
  "navatala_sparse_apply_compact_ids",
  "kernel:metal:navatala_sparse_apply_compact_ids",
  "abi-r1:metal:navatala_sparse_apply_compact_ids",
  "abi-r1:metal:navatala_sparse_apply_compact_ids",
  3,
  kAbiArgs_metal_navatala_sparse_apply_compact_ids
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_prolongation_f32[] = {
  { "aggregateId", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ProwPtr", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_prolongation_f32 = {
  1,
  "navatala_sparse_build_prolongation_f32",
  "metal",
  "navatala_sparse_build_prolongation_f32",
  "kernel:metal:navatala_sparse_build_prolongation_f32",
  "abi-r1:metal:navatala_sparse_build_prolongation_f32",
  "abi-r1:metal:navatala_sparse_build_prolongation_f32",
  5,
  kAbiArgs_metal_navatala_sparse_build_prolongation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_prolongation_terminator[] = {
  { "nRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ProwPtr", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_prolongation_terminator = {
  1,
  "navatala_sparse_build_prolongation_terminator",
  "metal",
  "navatala_sparse_build_prolongation_terminator",
  "kernel:metal:navatala_sparse_build_prolongation_terminator",
  "abi-r1:metal:navatala_sparse_build_prolongation_terminator",
  "abi-r1:metal:navatala_sparse_build_prolongation_terminator",
  2,
  kAbiArgs_metal_navatala_sparse_build_prolongation_terminator
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_spgemm_symbolic_row_nnz[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BrowPtr", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BcolIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsA", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "CrowNnz", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_spgemm_symbolic_row_nnz = {
  1,
  "navatala_sparse_spgemm_symbolic_row_nnz",
  "metal",
  "navatala_sparse_spgemm_symbolic_row_nnz",
  "kernel:metal:navatala_sparse_spgemm_symbolic_row_nnz",
  "abi-r1:metal:navatala_sparse_spgemm_symbolic_row_nnz",
  "abi-r1:metal:navatala_sparse_spgemm_symbolic_row_nnz",
  6,
  kAbiArgs_metal_navatala_sparse_spgemm_symbolic_row_nnz
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_spgemm_numeric_f32[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Avalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BrowPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BcolIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Bvalues", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "CrowPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsA", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "CcolIdx", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Cvalues", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_spgemm_numeric_f32 = {
  1,
  "navatala_sparse_spgemm_numeric_f32",
  "metal",
  "navatala_sparse_spgemm_numeric_f32",
  "kernel:metal:navatala_sparse_spgemm_numeric_f32",
  "abi-r1:metal:navatala_sparse_spgemm_numeric_f32",
  "abi-r1:metal:navatala_sparse_spgemm_numeric_f32",
  10,
  kAbiArgs_metal_navatala_sparse_spgemm_numeric_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_transpose[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCols", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "TrowPtr", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "TcolIdx", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Tvalues", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_transpose = {
  1,
  "navatala_sparse_csr_transpose",
  "metal",
  "navatala_sparse_csr_transpose",
  "kernel:metal:navatala_sparse_csr_transpose",
  "abi-r1:metal:navatala_sparse_csr_transpose",
  "abi-r1:metal:navatala_sparse_csr_transpose",
  8,
  kAbiArgs_metal_navatala_sparse_csr_transpose
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_adaptive_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degreeThreshold", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_adaptive_f32 = {
  1,
  "navatala_sparse_aggregate_adaptive_f32",
  "metal",
  "navatala_sparse_aggregate_adaptive_f32",
  "kernel:metal:navatala_sparse_aggregate_adaptive_f32",
  "abi-r1:metal:navatala_sparse_aggregate_adaptive_f32",
  "abi-r1:metal:navatala_sparse_aggregate_adaptive_f32",
  8,
  kAbiArgs_metal_navatala_sparse_aggregate_adaptive_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_multi_pairwise_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxPasses", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_multi_pairwise_f32 = {
  1,
  "navatala_sparse_aggregate_multi_pairwise_f32",
  "metal",
  "navatala_sparse_aggregate_multi_pairwise_f32",
  "kernel:metal:navatala_sparse_aggregate_multi_pairwise_f32",
  "abi-r1:metal:navatala_sparse_aggregate_multi_pairwise_f32",
  "abi-r1:metal:navatala_sparse_aggregate_multi_pairwise_f32",
  8,
  kAbiArgs_metal_navatala_sparse_aggregate_multi_pairwise_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_geometric_f32[] = {
  { "coordX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coordY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coordZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "radius", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_geometric_f32 = {
  1,
  "navatala_sparse_aggregate_geometric_f32",
  "metal",
  "navatala_sparse_aggregate_geometric_f32",
  "kernel:metal:navatala_sparse_aggregate_geometric_f32",
  "abi-r1:metal:navatala_sparse_aggregate_geometric_f32",
  "abi-r1:metal:navatala_sparse_aggregate_geometric_f32",
  7,
  kAbiArgs_metal_navatala_sparse_aggregate_geometric_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_serial_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_serial_f32 = {
  1,
  "navatala_sparse_aggregate_serial_f32",
  "metal",
  "navatala_sparse_aggregate_serial_f32",
  "kernel:metal:navatala_sparse_aggregate_serial_f32",
  "abi-r1:metal:navatala_sparse_aggregate_serial_f32",
  "abi-r1:metal:navatala_sparse_aggregate_serial_f32",
  7,
  kAbiArgs_metal_navatala_sparse_aggregate_serial_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_aggregate_low_degree_first_propose_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pickArray", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_aggregate_low_degree_first_propose_f32 = {
  1,
  "navatala_sparse_aggregate_low_degree_first_propose_f32",
  "metal",
  "navatala_sparse_aggregate_low_degree_first_propose_f32",
  "kernel:metal:navatala_sparse_aggregate_low_degree_first_propose_f32",
  "abi-r1:metal:navatala_sparse_aggregate_low_degree_first_propose_f32",
  "abi-r1:metal:navatala_sparse_aggregate_low_degree_first_propose_f32",
  6,
  kAbiArgs_metal_navatala_sparse_aggregate_low_degree_first_propose_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_assign_orphans_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nAggregates", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_assign_orphans_f32 = {
  1,
  "navatala_sparse_assign_orphans_f32",
  "metal",
  "navatala_sparse_assign_orphans_f32",
  "kernel:metal:navatala_sparse_assign_orphans_f32",
  "abi-r1:metal:navatala_sparse_assign_orphans_f32",
  "abi-r1:metal:navatala_sparse_assign_orphans_f32",
  6,
  kAbiArgs_metal_navatala_sparse_assign_orphans_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_spgemm_numeric_hash_f32[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Avalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BrowPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BcolIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Bvalues", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "CrowPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsA", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hashSize", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "CcolIdx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Cvalues", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_spgemm_numeric_hash_f32 = {
  1,
  "navatala_sparse_spgemm_numeric_hash_f32",
  "metal",
  "navatala_sparse_spgemm_numeric_hash_f32",
  "kernel:metal:navatala_sparse_spgemm_numeric_hash_f32",
  "abi-r1:metal:navatala_sparse_spgemm_numeric_hash_f32",
  "abi-r1:metal:navatala_sparse_spgemm_numeric_hash_f32",
  11,
  kAbiArgs_metal_navatala_sparse_spgemm_numeric_hash_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_spgemm_numeric_serial_f32[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Avalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BrowPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "BcolIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Bvalues", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "CrowPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRowsA", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "CcolIdx", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Cvalues", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_spgemm_numeric_serial_f32 = {
  1,
  "navatala_sparse_spgemm_numeric_serial_f32",
  "metal",
  "navatala_sparse_spgemm_numeric_serial_f32",
  "kernel:metal:navatala_sparse_spgemm_numeric_serial_f32",
  "abi-r1:metal:navatala_sparse_spgemm_numeric_serial_f32",
  "abi-r1:metal:navatala_sparse_spgemm_numeric_serial_f32",
  10,
  kAbiArgs_metal_navatala_sparse_spgemm_numeric_serial_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_smoothed_prolongation_f32[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Avalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "PsmoothValues", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_smoothed_prolongation_f32 = {
  1,
  "navatala_sparse_build_smoothed_prolongation_f32",
  "metal",
  "navatala_sparse_build_smoothed_prolongation_f32",
  "kernel:metal:navatala_sparse_build_smoothed_prolongation_f32",
  "abi-r1:metal:navatala_sparse_build_smoothed_prolongation_f32",
  "abi-r1:metal:navatala_sparse_build_smoothed_prolongation_f32",
  10,
  kAbiArgs_metal_navatala_sparse_build_smoothed_prolongation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_aggregate_sizes[] = {
  { "aggregateId", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nAggregates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggSizes", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_aggregate_sizes = {
  1,
  "navatala_sparse_compute_aggregate_sizes",
  "metal",
  "navatala_sparse_compute_aggregate_sizes",
  "kernel:metal:navatala_sparse_compute_aggregate_sizes",
  "abi-r1:metal:navatala_sparse_compute_aggregate_sizes",
  "abi-r1:metal:navatala_sparse_compute_aggregate_sizes",
  4,
  kAbiArgs_metal_navatala_sparse_compute_aggregate_sizes
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_coarsening_ratio[] = {
  { "nFine", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCoarse", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ratio", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_coarsening_ratio = {
  1,
  "navatala_sparse_compute_coarsening_ratio",
  "metal",
  "navatala_sparse_compute_coarsening_ratio",
  "kernel:metal:navatala_sparse_compute_coarsening_ratio",
  "abi-r1:metal:navatala_sparse_compute_coarsening_ratio",
  "abi-r1:metal:navatala_sparse_compute_coarsening_ratio",
  3,
  kAbiArgs_metal_navatala_sparse_compute_coarsening_ratio
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_p_m_i_s[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_p_m_i_s = {
  1,
  "navatala_sparse_cf_split_p_m_i_s",
  "metal",
  "navatala_sparse_cf_split_p_m_i_s",
  "kernel:metal:navatala_sparse_cf_split_p_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_p_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_p_m_i_s",
  6,
  kAbiArgs_metal_navatala_sparse_cf_split_p_m_i_s
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_h_m_i_s[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_h_m_i_s = {
  1,
  "navatala_sparse_cf_split_h_m_i_s",
  "metal",
  "navatala_sparse_cf_split_h_m_i_s",
  "kernel:metal:navatala_sparse_cf_split_h_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_h_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_h_m_i_s",
  6,
  kAbiArgs_metal_navatala_sparse_cf_split_h_m_i_s
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_h_m_i_s_phase2_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_h_m_i_s_phase2_f32 = {
  1,
  "navatala_sparse_cf_split_h_m_i_s_phase2_f32",
  "metal",
  "navatala_sparse_cf_split_h_m_i_s_phase2_f32",
  "kernel:metal:navatala_sparse_cf_split_h_m_i_s_phase2_f32",
  "abi-r1:metal:navatala_sparse_cf_split_h_m_i_s_phase2_f32",
  "abi-r1:metal:navatala_sparse_cf_split_h_m_i_s_phase2_f32",
  5,
  kAbiArgs_metal_navatala_sparse_cf_split_h_m_i_s_phase2_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_r_s[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_r_s = {
  1,
  "navatala_sparse_cf_split_r_s",
  "metal",
  "navatala_sparse_cf_split_r_s",
  "kernel:metal:navatala_sparse_cf_split_r_s",
  "abi-r1:metal:navatala_sparse_cf_split_r_s",
  "abi-r1:metal:navatala_sparse_cf_split_r_s",
  5,
  kAbiArgs_metal_navatala_sparse_cf_split_r_s
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_distance1_count_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_distance1_count_f32 = {
  1,
  "navatala_sparse_interpolate_distance1_count_f32",
  "metal",
  "navatala_sparse_interpolate_distance1_count_f32",
  "kernel:metal:navatala_sparse_interpolate_distance1_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance1_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance1_count_f32",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_distance1_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_distance1_count_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_distance1_count_f64 = {
  1,
  "navatala_sparse_interpolate_distance1_count_f64",
  "metal",
  "navatala_sparse_interpolate_distance1_count_f64",
  "kernel:metal:navatala_sparse_interpolate_distance1_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_distance1_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_distance1_count_f64",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_distance1_count_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_distance1_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "PcolIdx", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_distance1_f32 = {
  1,
  "navatala_sparse_interpolate_distance1_f32",
  "metal",
  "navatala_sparse_interpolate_distance1_f32",
  "kernel:metal:navatala_sparse_interpolate_distance1_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance1_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance1_f32",
  9,
  kAbiArgs_metal_navatala_sparse_interpolate_distance1_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_distance2_count_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_distance2_count_f32 = {
  1,
  "navatala_sparse_interpolate_distance2_count_f32",
  "metal",
  "navatala_sparse_interpolate_distance2_count_f32",
  "kernel:metal:navatala_sparse_interpolate_distance2_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance2_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance2_count_f32",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_distance2_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_distance2_count_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_distance2_count_f64 = {
  1,
  "navatala_sparse_interpolate_distance2_count_f64",
  "metal",
  "navatala_sparse_interpolate_distance2_count_f64",
  "kernel:metal:navatala_sparse_interpolate_distance2_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_distance2_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_distance2_count_f64",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_distance2_count_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_distance2_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "PcolIdx", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_distance2_f32 = {
  1,
  "navatala_sparse_interpolate_distance2_f32",
  "metal",
  "navatala_sparse_interpolate_distance2_f32",
  "kernel:metal:navatala_sparse_interpolate_distance2_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance2_f32",
  "abi-r1:metal:navatala_sparse_interpolate_distance2_f32",
  10,
  kAbiArgs_metal_navatala_sparse_interpolate_distance2_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_truncate_interpolation_count_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "threshold", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRowOut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_truncate_interpolation_count_f32 = {
  1,
  "navatala_sparse_truncate_interpolation_count_f32",
  "metal",
  "navatala_sparse_truncate_interpolation_count_f32",
  "kernel:metal:navatala_sparse_truncate_interpolation_count_f32",
  "abi-r1:metal:navatala_sparse_truncate_interpolation_count_f32",
  "abi-r1:metal:navatala_sparse_truncate_interpolation_count_f32",
  6,
  kAbiArgs_metal_navatala_sparse_truncate_interpolation_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_truncate_interpolation_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "threshold", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ProwPtrOut", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdxOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PvaluesOut", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_truncate_interpolation_f32 = {
  1,
  "navatala_sparse_truncate_interpolation_f32",
  "metal",
  "navatala_sparse_truncate_interpolation_f32",
  "kernel:metal:navatala_sparse_truncate_interpolation_f32",
  "abi-r1:metal:navatala_sparse_truncate_interpolation_f32",
  "abi-r1:metal:navatala_sparse_truncate_interpolation_f32",
  8,
  kAbiArgs_metal_navatala_sparse_truncate_interpolation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_aggressive_p_m_i_s[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarkingPrev", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_aggressive_p_m_i_s = {
  1,
  "navatala_sparse_cf_split_aggressive_p_m_i_s",
  "metal",
  "navatala_sparse_cf_split_aggressive_p_m_i_s",
  "kernel:metal:navatala_sparse_cf_split_aggressive_p_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_aggressive_p_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_aggressive_p_m_i_s",
  7,
  kAbiArgs_metal_navatala_sparse_cf_split_aggressive_p_m_i_s
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_aggressive_h_m_i_s[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarkingPrev", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_aggressive_h_m_i_s = {
  1,
  "navatala_sparse_cf_split_aggressive_h_m_i_s",
  "metal",
  "navatala_sparse_cf_split_aggressive_h_m_i_s",
  "kernel:metal:navatala_sparse_cf_split_aggressive_h_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_aggressive_h_m_i_s",
  "abi-r1:metal:navatala_sparse_cf_split_aggressive_h_m_i_s",
  7,
  kAbiArgs_metal_navatala_sparse_cf_split_aggressive_h_m_i_s
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_c_r_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "crThreshold", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_c_r_f32 = {
  1,
  "navatala_sparse_cf_split_c_r_f32",
  "metal",
  "navatala_sparse_cf_split_c_r_f32",
  "kernel:metal:navatala_sparse_cf_split_c_r_f32",
  "abi-r1:metal:navatala_sparse_cf_split_c_r_f32",
  "abi-r1:metal:navatala_sparse_cf_split_c_r_f32",
  7,
  kAbiArgs_metal_navatala_sparse_cf_split_c_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_cf_split_c_l_j_p[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cfMarking", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_cf_split_c_l_j_p = {
  1,
  "navatala_sparse_cf_split_c_l_j_p",
  "metal",
  "navatala_sparse_cf_split_c_l_j_p",
  "kernel:metal:navatala_sparse_cf_split_c_l_j_p",
  "abi-r1:metal:navatala_sparse_cf_split_c_l_j_p",
  "abi-r1:metal:navatala_sparse_cf_split_c_l_j_p",
  7,
  kAbiArgs_metal_navatala_sparse_cf_split_c_l_j_p
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_multipass_count_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_multipass_count_f32 = {
  1,
  "navatala_sparse_interpolate_multipass_count_f32",
  "metal",
  "navatala_sparse_interpolate_multipass_count_f32",
  "kernel:metal:navatala_sparse_interpolate_multipass_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_multipass_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_multipass_count_f32",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_multipass_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_multipass_count_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_multipass_count_f64 = {
  1,
  "navatala_sparse_interpolate_multipass_count_f64",
  "metal",
  "navatala_sparse_interpolate_multipass_count_f64",
  "kernel:metal:navatala_sparse_interpolate_multipass_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_multipass_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_multipass_count_f64",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_multipass_count_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_multipass_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxPass", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "PcolIdx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_multipass_f32 = {
  1,
  "navatala_sparse_interpolate_multipass_f32",
  "metal",
  "navatala_sparse_interpolate_multipass_f32",
  "kernel:metal:navatala_sparse_interpolate_multipass_f32",
  "abi-r1:metal:navatala_sparse_interpolate_multipass_f32",
  "abi-r1:metal:navatala_sparse_interpolate_multipass_f32",
  11,
  kAbiArgs_metal_navatala_sparse_interpolate_multipass_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_strength_symmetric_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "strongMask", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_strength_symmetric_f32 = {
  1,
  "navatala_sparse_strength_symmetric_f32",
  "metal",
  "navatala_sparse_strength_symmetric_f32",
  "kernel:metal:navatala_sparse_strength_symmetric_f32",
  "abi-r1:metal:navatala_sparse_strength_symmetric_f32",
  "abi-r1:metal:navatala_sparse_strength_symmetric_f32",
  7,
  kAbiArgs_metal_navatala_sparse_strength_symmetric_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_strength_algebraic_dist_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "testVectors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nTestVectors", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "theta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "strongMask", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_strength_algebraic_dist_f32 = {
  1,
  "navatala_sparse_strength_algebraic_dist_f32",
  "metal",
  "navatala_sparse_strength_algebraic_dist_f32",
  "kernel:metal:navatala_sparse_strength_algebraic_dist_f32",
  "abi-r1:metal:navatala_sparse_strength_algebraic_dist_f32",
  "abi-r1:metal:navatala_sparse_strength_algebraic_dist_f32",
  8,
  kAbiArgs_metal_navatala_sparse_strength_algebraic_dist_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_strength_affinity_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "strengthValues", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_strength_affinity_f32 = {
  1,
  "navatala_sparse_strength_affinity_f32",
  "metal",
  "navatala_sparse_strength_affinity_f32",
  "kernel:metal:navatala_sparse_strength_affinity_f32",
  "abi-r1:metal:navatala_sparse_strength_affinity_f32",
  "abi-r1:metal:navatala_sparse_strength_affinity_f32",
  7,
  kAbiArgs_metal_navatala_sparse_strength_affinity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_count_c_f_points[] = {
  { "cfMarking", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCoarse", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFine", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_count_c_f_points = {
  1,
  "navatala_sparse_count_c_f_points",
  "metal",
  "navatala_sparse_count_c_f_points",
  "kernel:metal:navatala_sparse_count_c_f_points",
  "abi-r1:metal:navatala_sparse_count_c_f_points",
  "abi-r1:metal:navatala_sparse_count_c_f_points",
  4,
  kAbiArgs_metal_navatala_sparse_count_c_f_points
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_grid_complexity[] = {
  { "levelNnz", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nLevels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gridComplexity", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_grid_complexity = {
  1,
  "navatala_sparse_compute_grid_complexity",
  "metal",
  "navatala_sparse_compute_grid_complexity",
  "kernel:metal:navatala_sparse_compute_grid_complexity",
  "abi-r1:metal:navatala_sparse_compute_grid_complexity",
  "abi-r1:metal:navatala_sparse_compute_grid_complexity",
  3,
  kAbiArgs_metal_navatala_sparse_compute_grid_complexity
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_operator_complexity[] = {
  { "levelNnz", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "levelRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nLevels", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "operatorComplexity", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_operator_complexity = {
  1,
  "navatala_sparse_compute_operator_complexity",
  "metal",
  "navatala_sparse_compute_operator_complexity",
  "kernel:metal:navatala_sparse_compute_operator_complexity",
  "abi-r1:metal:navatala_sparse_compute_operator_complexity",
  "abi-r1:metal:navatala_sparse_compute_operator_complexity",
  4,
  kAbiArgs_metal_navatala_sparse_compute_operator_complexity
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_ext_no_i_count_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_ext_no_i_count_f32 = {
  1,
  "navatala_sparse_interpolate_ext_no_i_count_f32",
  "metal",
  "navatala_sparse_interpolate_ext_no_i_count_f32",
  "kernel:metal:navatala_sparse_interpolate_ext_no_i_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_ext_no_i_count_f32",
  "abi-r1:metal:navatala_sparse_interpolate_ext_no_i_count_f32",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_ext_no_i_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_ext_no_i_count_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_ext_no_i_count_f64 = {
  1,
  "navatala_sparse_interpolate_ext_no_i_count_f64",
  "metal",
  "navatala_sparse_interpolate_ext_no_i_count_f64",
  "kernel:metal:navatala_sparse_interpolate_ext_no_i_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_ext_no_i_count_f64",
  "abi-r1:metal:navatala_sparse_interpolate_ext_no_i_count_f64",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_ext_no_i_count_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_ext_no_i_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "PcolIdx", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_ext_no_i_f32 = {
  1,
  "navatala_sparse_interpolate_ext_no_i_f32",
  "metal",
  "navatala_sparse_interpolate_ext_no_i_f32",
  "kernel:metal:navatala_sparse_interpolate_ext_no_i_f32",
  "abi-r1:metal:navatala_sparse_interpolate_ext_no_i_f32",
  "abi-r1:metal:navatala_sparse_interpolate_ext_no_i_f32",
  10,
  kAbiArgs_metal_navatala_sparse_interpolate_ext_no_i_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_f_f_weights_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Pvalues", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_f_f_weights_f32 = {
  1,
  "navatala_sparse_interpolate_f_f_weights_f32",
  "metal",
  "navatala_sparse_interpolate_f_f_weights_f32",
  "kernel:metal:navatala_sparse_interpolate_f_f_weights_f32",
  "abi-r1:metal:navatala_sparse_interpolate_f_f_weights_f32",
  "abi-r1:metal:navatala_sparse_interpolate_f_f_weights_f32",
  10,
  kAbiArgs_metal_navatala_sparse_interpolate_f_f_weights_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_symbolic_row_nnz_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ProwNnz", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_symbolic_row_nnz_f32 = {
  1,
  "navatala_sparse_interpolate_symbolic_row_nnz_f32",
  "metal",
  "navatala_sparse_interpolate_symbolic_row_nnz_f32",
  "kernel:metal:navatala_sparse_interpolate_symbolic_row_nnz_f32",
  "abi-r1:metal:navatala_sparse_interpolate_symbolic_row_nnz_f32",
  "abi-r1:metal:navatala_sparse_interpolate_symbolic_row_nnz_f32",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_symbolic_row_nnz_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_interpolate_symbolic_row_nnz_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfMarking", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ProwNnz", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_interpolate_symbolic_row_nnz_f64 = {
  1,
  "navatala_sparse_interpolate_symbolic_row_nnz_f64",
  "metal",
  "navatala_sparse_interpolate_symbolic_row_nnz_f64",
  "kernel:metal:navatala_sparse_interpolate_symbolic_row_nnz_f64",
  "abi-r1:metal:navatala_sparse_interpolate_symbolic_row_nnz_f64",
  "abi-r1:metal:navatala_sparse_interpolate_symbolic_row_nnz_f64",
  6,
  kAbiArgs_metal_navatala_sparse_interpolate_symbolic_row_nnz_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_coarse_to_fine_map[] = {
  { "cfMarking", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseOffset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseToFine", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_coarse_to_fine_map = {
  1,
  "navatala_sparse_build_coarse_to_fine_map",
  "metal",
  "navatala_sparse_build_coarse_to_fine_map",
  "kernel:metal:navatala_sparse_build_coarse_to_fine_map",
  "abi-r1:metal:navatala_sparse_build_coarse_to_fine_map",
  "abi-r1:metal:navatala_sparse_build_coarse_to_fine_map",
  4,
  kAbiArgs_metal_navatala_sparse_build_coarse_to_fine_map
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_fine_to_coarse_map_count[] = {
  { "cfMarking", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseFlag", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_fine_to_coarse_map_count = {
  1,
  "navatala_sparse_build_fine_to_coarse_map_count",
  "metal",
  "navatala_sparse_build_fine_to_coarse_map_count",
  "kernel:metal:navatala_sparse_build_fine_to_coarse_map_count",
  "abi-r1:metal:navatala_sparse_build_fine_to_coarse_map_count",
  "abi-r1:metal:navatala_sparse_build_fine_to_coarse_map_count",
  3,
  kAbiArgs_metal_navatala_sparse_build_fine_to_coarse_map_count
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_fine_to_coarse_map[] = {
  { "cfMarking", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseOffset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "fineToCoarse", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_fine_to_coarse_map = {
  1,
  "navatala_sparse_build_fine_to_coarse_map",
  "metal",
  "navatala_sparse_build_fine_to_coarse_map",
  "kernel:metal:navatala_sparse_build_fine_to_coarse_map",
  "abi-r1:metal:navatala_sparse_build_fine_to_coarse_map",
  "abi-r1:metal:navatala_sparse_build_fine_to_coarse_map",
  4,
  kAbiArgs_metal_navatala_sparse_build_fine_to_coarse_map
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_em_initial_prolongation_f32[] = {
  { "aggregateId", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ProwPtr", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_em_initial_prolongation_f32 = {
  1,
  "navatala_sparse_em_initial_prolongation_f32",
  "metal",
  "navatala_sparse_em_initial_prolongation_f32",
  "kernel:metal:navatala_sparse_em_initial_prolongation_f32",
  "abi-r1:metal:navatala_sparse_em_initial_prolongation_f32",
  "abi-r1:metal:navatala_sparse_em_initial_prolongation_f32",
  5,
  kAbiArgs_metal_navatala_sparse_em_initial_prolongation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_em_energy_gradient_f32[] = {
  { "ArowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Avalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ProwPtr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradient", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_em_energy_gradient_f32 = {
  1,
  "navatala_sparse_em_energy_gradient_f32",
  "metal",
  "navatala_sparse_em_energy_gradient_f32",
  "kernel:metal:navatala_sparse_em_energy_gradient_f32",
  "abi-r1:metal:navatala_sparse_em_energy_gradient_f32",
  "abi-r1:metal:navatala_sparse_em_energy_gradient_f32",
  8,
  kAbiArgs_metal_navatala_sparse_em_energy_gradient_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_em_update_prolongation_f32[] = {
  { "gradient", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "stepSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzP", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Pvalues", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_em_update_prolongation_f32 = {
  1,
  "navatala_sparse_em_update_prolongation_f32",
  "metal",
  "navatala_sparse_em_update_prolongation_f32",
  "kernel:metal:navatala_sparse_em_update_prolongation_f32",
  "abi-r1:metal:navatala_sparse_em_update_prolongation_f32",
  "abi-r1:metal:navatala_sparse_em_update_prolongation_f32",
  4,
  kAbiArgs_metal_navatala_sparse_em_update_prolongation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_em_normalize_p_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_em_normalize_p_f32 = {
  1,
  "navatala_sparse_em_normalize_p_f32",
  "metal",
  "navatala_sparse_em_normalize_p_f32",
  "kernel:metal:navatala_sparse_em_normalize_p_f32",
  "abi-r1:metal:navatala_sparse_em_normalize_p_f32",
  "abi-r1:metal:navatala_sparse_em_normalize_p_f32",
  3,
  kAbiArgs_metal_navatala_sparse_em_normalize_p_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_em_energy_eval_partials_f32[] = {
  { "APvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nnzP", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "denPartials", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_em_energy_eval_partials_f32 = {
  1,
  "navatala_sparse_em_energy_eval_partials_f32",
  "metal",
  "navatala_sparse_em_energy_eval_partials_f32",
  "kernel:metal:navatala_sparse_em_energy_eval_partials_f32",
  "abi-r1:metal:navatala_sparse_em_energy_eval_partials_f32",
  "abi-r1:metal:navatala_sparse_em_energy_eval_partials_f32",
  5,
  kAbiArgs_metal_navatala_sparse_em_energy_eval_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_parallel_greedy_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_parallel_greedy_coloring = {
  1,
  "navatala_sparse_parallel_greedy_coloring",
  "metal",
  "navatala_sparse_parallel_greedy_coloring",
  "kernel:metal:navatala_sparse_parallel_greedy_coloring",
  "abi-r1:metal:navatala_sparse_parallel_greedy_coloring",
  "abi-r1:metal:navatala_sparse_parallel_greedy_coloring",
  6,
  kAbiArgs_metal_navatala_sparse_parallel_greedy_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_min_max_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_min_max_coloring = {
  1,
  "navatala_sparse_min_max_coloring",
  "metal",
  "navatala_sparse_min_max_coloring",
  "kernel:metal:navatala_sparse_min_max_coloring",
  "abi-r1:metal:navatala_sparse_min_max_coloring",
  "abi-r1:metal:navatala_sparse_min_max_coloring",
  5,
  kAbiArgs_metal_navatala_sparse_min_max_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_min_max2_ring_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_min_max2_ring_coloring = {
  1,
  "navatala_sparse_min_max2_ring_coloring",
  "metal",
  "navatala_sparse_min_max2_ring_coloring",
  "kernel:metal:navatala_sparse_min_max2_ring_coloring",
  "abi-r1:metal:navatala_sparse_min_max2_ring_coloring",
  "abi-r1:metal:navatala_sparse_min_max2_ring_coloring",
  5,
  kAbiArgs_metal_navatala_sparse_min_max2_ring_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multi_hash_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nHashFunctions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multi_hash_coloring = {
  1,
  "navatala_sparse_multi_hash_coloring",
  "metal",
  "navatala_sparse_multi_hash_coloring",
  "kernel:metal:navatala_sparse_multi_hash_coloring",
  "abi-r1:metal:navatala_sparse_multi_hash_coloring",
  "abi-r1:metal:navatala_sparse_multi_hash_coloring",
  6,
  kAbiArgs_metal_navatala_sparse_multi_hash_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_build_color_offsets[] = {
  { "colors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nColors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colorOffsets", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reorderPerm", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_build_color_offsets = {
  1,
  "navatala_sparse_build_color_offsets",
  "metal",
  "navatala_sparse_build_color_offsets",
  "kernel:metal:navatala_sparse_build_color_offsets",
  "abi-r1:metal:navatala_sparse_build_color_offsets",
  "abi-r1:metal:navatala_sparse_build_color_offsets",
  5,
  kAbiArgs_metal_navatala_sparse_build_color_offsets
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_reorder_c_s_r_by_color[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reorderPerm", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newRowPtr", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newColIdx", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newValues", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_reorder_c_s_r_by_color = {
  1,
  "navatala_sparse_reorder_c_s_r_by_color",
  "metal",
  "navatala_sparse_reorder_c_s_r_by_color",
  "kernel:metal:navatala_sparse_reorder_c_s_r_by_color",
  "abi-r1:metal:navatala_sparse_reorder_c_s_r_by_color",
  "abi-r1:metal:navatala_sparse_reorder_c_s_r_by_color",
  8,
  kAbiArgs_metal_navatala_sparse_reorder_c_s_r_by_color
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_round_robin_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nTargetColors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_round_robin_coloring = {
  1,
  "navatala_sparse_round_robin_coloring",
  "metal",
  "navatala_sparse_round_robin_coloring",
  "kernel:metal:navatala_sparse_round_robin_coloring",
  "abi-r1:metal:navatala_sparse_round_robin_coloring",
  "abi-r1:metal:navatala_sparse_round_robin_coloring",
  5,
  kAbiArgs_metal_navatala_sparse_round_robin_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_bfs_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "startRow", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_bfs_coloring = {
  1,
  "navatala_sparse_bfs_coloring",
  "metal",
  "navatala_sparse_bfs_coloring",
  "kernel:metal:navatala_sparse_bfs_coloring",
  "abi-r1:metal:navatala_sparse_bfs_coloring",
  "abi-r1:metal:navatala_sparse_bfs_coloring",
  6,
  kAbiArgs_metal_navatala_sparse_bfs_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_locally_downwind_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_locally_downwind_coloring = {
  1,
  "navatala_sparse_locally_downwind_coloring",
  "metal",
  "navatala_sparse_locally_downwind_coloring",
  "kernel:metal:navatala_sparse_locally_downwind_coloring",
  "abi-r1:metal:navatala_sparse_locally_downwind_coloring",
  "abi-r1:metal:navatala_sparse_locally_downwind_coloring",
  6,
  kAbiArgs_metal_navatala_sparse_locally_downwind_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_greedy_recolor_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nConflicts", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_greedy_recolor_f32 = {
  1,
  "navatala_sparse_greedy_recolor_f32",
  "metal",
  "navatala_sparse_greedy_recolor_f32",
  "kernel:metal:navatala_sparse_greedy_recolor_f32",
  "abi-r1:metal:navatala_sparse_greedy_recolor_f32",
  "abi-r1:metal:navatala_sparse_greedy_recolor_f32",
  6,
  kAbiArgs_metal_navatala_sparse_greedy_recolor_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_greedy_recolor_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nConflicts", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_greedy_recolor_f64 = {
  1,
  "navatala_sparse_greedy_recolor_f64",
  "metal",
  "navatala_sparse_greedy_recolor_f64",
  "kernel:metal:navatala_sparse_greedy_recolor_f64",
  "abi-r1:metal:navatala_sparse_greedy_recolor_f64",
  "abi-r1:metal:navatala_sparse_greedy_recolor_f64",
  6,
  kAbiArgs_metal_navatala_sparse_greedy_recolor_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_validate_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nConflicts", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "conflictFlags", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_validate_coloring = {
  1,
  "navatala_sparse_validate_coloring",
  "metal",
  "navatala_sparse_validate_coloring",
  "kernel:metal:navatala_sparse_validate_coloring",
  "abi-r1:metal:navatala_sparse_validate_coloring",
  "abi-r1:metal:navatala_sparse_validate_coloring",
  6,
  kAbiArgs_metal_navatala_sparse_validate_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_color_histogram[] = {
  { "colors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nColors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_color_histogram = {
  1,
  "navatala_sparse_color_histogram",
  "metal",
  "navatala_sparse_color_histogram",
  "kernel:metal:navatala_sparse_color_histogram",
  "abi-r1:metal:navatala_sparse_color_histogram",
  "abi-r1:metal:navatala_sparse_color_histogram",
  4,
  kAbiArgs_metal_navatala_sparse_color_histogram
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_uniform_random_coloring[] = {
  { "nRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nTargetColors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_uniform_random_coloring = {
  1,
  "navatala_sparse_uniform_random_coloring",
  "metal",
  "navatala_sparse_uniform_random_coloring",
  "kernel:metal:navatala_sparse_uniform_random_coloring",
  "abi-r1:metal:navatala_sparse_uniform_random_coloring",
  "abi-r1:metal:navatala_sparse_uniform_random_coloring",
  4,
  kAbiArgs_metal_navatala_sparse_uniform_random_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_saturation_coloring[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colors", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nColors", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_saturation_coloring = {
  1,
  "navatala_sparse_saturation_coloring",
  "metal",
  "navatala_sparse_saturation_coloring",
  "kernel:metal:navatala_sparse_saturation_coloring",
  "abi-r1:metal:navatala_sparse_saturation_coloring",
  "abi-r1:metal:navatala_sparse_saturation_coloring",
  5,
  kAbiArgs_metal_navatala_sparse_saturation_coloring
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_reorder_vector_by_color_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reorderPerm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xReordered", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_reorder_vector_by_color_f32 = {
  1,
  "navatala_sparse_reorder_vector_by_color_f32",
  "metal",
  "navatala_sparse_reorder_vector_by_color_f32",
  "kernel:metal:navatala_sparse_reorder_vector_by_color_f32",
  "abi-r1:metal:navatala_sparse_reorder_vector_by_color_f32",
  "abi-r1:metal:navatala_sparse_reorder_vector_by_color_f32",
  4,
  kAbiArgs_metal_navatala_sparse_reorder_vector_by_color_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_diag_inv_sqrt_f32[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dinvsqrt", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_diag_inv_sqrt_f32 = {
  1,
  "navatala_sparse_diag_inv_sqrt_f32",
  "metal",
  "navatala_sparse_diag_inv_sqrt_f32",
  "kernel:metal:navatala_sparse_diag_inv_sqrt_f32",
  "abi-r1:metal:navatala_sparse_diag_inv_sqrt_f32",
  "abi-r1:metal:navatala_sparse_diag_inv_sqrt_f32",
  3,
  kAbiArgs_metal_navatala_sparse_diag_inv_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_apply_diag_sym_scale_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dinvsqrt", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_apply_diag_sym_scale_f32 = {
  1,
  "navatala_sparse_apply_diag_sym_scale_f32",
  "metal",
  "navatala_sparse_apply_diag_sym_scale_f32",
  "kernel:metal:navatala_sparse_apply_diag_sym_scale_f32",
  "abi-r1:metal:navatala_sparse_apply_diag_sym_scale_f32",
  "abi-r1:metal:navatala_sparse_apply_diag_sym_scale_f32",
  5,
  kAbiArgs_metal_navatala_sparse_apply_diag_sym_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_binorm_row_scale_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowScale", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_binorm_row_scale_f32 = {
  1,
  "navatala_sparse_binorm_row_scale_f32",
  "metal",
  "navatala_sparse_binorm_row_scale_f32",
  "kernel:metal:navatala_sparse_binorm_row_scale_f32",
  "abi-r1:metal:navatala_sparse_binorm_row_scale_f32",
  "abi-r1:metal:navatala_sparse_binorm_row_scale_f32",
  4,
  kAbiArgs_metal_navatala_sparse_binorm_row_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_shifted_sp_m_v_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_shifted_sp_m_v_f32 = {
  1,
  "navatala_sparse_shifted_sp_m_v_f32",
  "metal",
  "navatala_sparse_shifted_sp_m_v_f32",
  "kernel:metal:navatala_sparse_shifted_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_shifted_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_shifted_sp_m_v_f32",
  7,
  kAbiArgs_metal_navatala_sparse_shifted_sp_m_v_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_deflated_project_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Z", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ZtX", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "w", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_deflated_project_f32 = {
  1,
  "navatala_sparse_deflated_project_f32",
  "metal",
  "navatala_sparse_deflated_project_f32",
  "kernel:metal:navatala_sparse_deflated_project_f32",
  "abi-r1:metal:navatala_sparse_deflated_project_f32",
  "abi-r1:metal:navatala_sparse_deflated_project_f32",
  6,
  kAbiArgs_metal_navatala_sparse_deflated_project_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_page_rank_iter_f32[] = {
  { "Ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "damping", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_page_rank_iter_f32 = {
  1,
  "navatala_sparse_page_rank_iter_f32",
  "metal",
  "navatala_sparse_page_rank_iter_f32",
  "kernel:metal:navatala_sparse_page_rank_iter_f32",
  "abi-r1:metal:navatala_sparse_page_rank_iter_f32",
  "abi-r1:metal:navatala_sparse_page_rank_iter_f32",
  4,
  kAbiArgs_metal_navatala_sparse_page_rank_iter_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_w_cycle_weight_partials_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "e", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ae", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rePartials", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AeePartials", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_w_cycle_weight_partials_f32 = {
  1,
  "navatala_sparse_w_cycle_weight_partials_f32",
  "metal",
  "navatala_sparse_w_cycle_weight_partials_f32",
  "kernel:metal:navatala_sparse_w_cycle_weight_partials_f32",
  "abi-r1:metal:navatala_sparse_w_cycle_weight_partials_f32",
  "abi-r1:metal:navatala_sparse_w_cycle_weight_partials_f32",
  6,
  kAbiArgs_metal_navatala_sparse_w_cycle_weight_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_convergence_factor_f32[] = {
  { "normCurrent", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "normPrevious", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rho", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_convergence_factor_f32 = {
  1,
  "navatala_sparse_convergence_factor_f32",
  "metal",
  "navatala_sparse_convergence_factor_f32",
  "kernel:metal:navatala_sparse_convergence_factor_f32",
  "abi-r1:metal:navatala_sparse_convergence_factor_f32",
  "abi-r1:metal:navatala_sparse_convergence_factor_f32",
  3,
  kAbiArgs_metal_navatala_sparse_convergence_factor_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_block_gram_schmidt_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "j", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Xj", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_block_gram_schmidt_f32 = {
  1,
  "navatala_sparse_block_gram_schmidt_f32",
  "metal",
  "navatala_sparse_block_gram_schmidt_f32",
  "kernel:metal:navatala_sparse_block_gram_schmidt_f32",
  "abi-r1:metal:navatala_sparse_block_gram_schmidt_f32",
  "abi-r1:metal:navatala_sparse_block_gram_schmidt_f32",
  6,
  kAbiArgs_metal_navatala_sparse_block_gram_schmidt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_small_dense_q_r_f32[] = {
  { "m", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tau", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_small_dense_q_r_f32 = {
  1,
  "navatala_sparse_small_dense_q_r_f32",
  "metal",
  "navatala_sparse_small_dense_q_r_f32",
  "kernel:metal:navatala_sparse_small_dense_q_r_f32",
  "abi-r1:metal:navatala_sparse_small_dense_q_r_f32",
  "abi-r1:metal:navatala_sparse_small_dense_q_r_f32",
  4,
  kAbiArgs_metal_navatala_sparse_small_dense_q_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_lanczos_step_f32[] = {
  { "Av", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vCurrent", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vPrevious", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vNext", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "betaNext", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_lanczos_step_f32 = {
  1,
  "navatala_sparse_lanczos_step_f32",
  "metal",
  "navatala_sparse_lanczos_step_f32",
  "kernel:metal:navatala_sparse_lanczos_step_f32",
  "abi-r1:metal:navatala_sparse_lanczos_step_f32",
  "abi-r1:metal:navatala_sparse_lanczos_step_f32",
  8,
  kAbiArgs_metal_navatala_sparse_lanczos_step_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_arnoldi_step_f32[] = {
  { "Av", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "V", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hCol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "j", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vNext", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hNextJ", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_arnoldi_step_f32 = {
  1,
  "navatala_sparse_arnoldi_step_f32",
  "metal",
  "navatala_sparse_arnoldi_step_f32",
  "kernel:metal:navatala_sparse_arnoldi_step_f32",
  "abi-r1:metal:navatala_sparse_arnoldi_step_f32",
  "abi-r1:metal:navatala_sparse_arnoldi_step_f32",
  7,
  kAbiArgs_metal_navatala_sparse_arnoldi_step_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_lobpcg_rayleigh_ritz_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "AX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "XtAX", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "XtX", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_lobpcg_rayleigh_ritz_f32 = {
  1,
  "navatala_sparse_lobpcg_rayleigh_ritz_f32",
  "metal",
  "navatala_sparse_lobpcg_rayleigh_ritz_f32",
  "kernel:metal:navatala_sparse_lobpcg_rayleigh_ritz_f32",
  "abi-r1:metal:navatala_sparse_lobpcg_rayleigh_ritz_f32",
  "abi-r1:metal:navatala_sparse_lobpcg_rayleigh_ritz_f32",
  6,
  kAbiArgs_metal_navatala_sparse_lobpcg_rayleigh_ritz_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_lobpcg_residual_f32[] = {
  { "AX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdas", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "R", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_lobpcg_residual_f32 = {
  1,
  "navatala_sparse_lobpcg_residual_f32",
  "metal",
  "navatala_sparse_lobpcg_residual_f32",
  "kernel:metal:navatala_sparse_lobpcg_residual_f32",
  "abi-r1:metal:navatala_sparse_lobpcg_residual_f32",
  "abi-r1:metal:navatala_sparse_lobpcg_residual_f32",
  6,
  kAbiArgs_metal_navatala_sparse_lobpcg_residual_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_power_iteration_normalize_f32[] = {
  { "Av", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "norm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_power_iteration_normalize_f32 = {
  1,
  "navatala_sparse_power_iteration_normalize_f32",
  "metal",
  "navatala_sparse_power_iteration_normalize_f32",
  "kernel:metal:navatala_sparse_power_iteration_normalize_f32",
  "abi-r1:metal:navatala_sparse_power_iteration_normalize_f32",
  "abi-r1:metal:navatala_sparse_power_iteration_normalize_f32",
  4,
  kAbiArgs_metal_navatala_sparse_power_iteration_normalize_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_rayleigh_quotient_partials_f32[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Av", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vAvPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vvPartials", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_rayleigh_quotient_partials_f32 = {
  1,
  "navatala_sparse_rayleigh_quotient_partials_f32",
  "metal",
  "navatala_sparse_rayleigh_quotient_partials_f32",
  "kernel:metal:navatala_sparse_rayleigh_quotient_partials_f32",
  "abi-r1:metal:navatala_sparse_rayleigh_quotient_partials_f32",
  "abi-r1:metal:navatala_sparse_rayleigh_quotient_partials_f32",
  5,
  kAbiArgs_metal_navatala_sparse_rayleigh_quotient_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_jacobi_davidson_correction_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "t", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_jacobi_davidson_correction_f32 = {
  1,
  "navatala_sparse_jacobi_davidson_correction_f32",
  "metal",
  "navatala_sparse_jacobi_davidson_correction_f32",
  "kernel:metal:navatala_sparse_jacobi_davidson_correction_f32",
  "abi-r1:metal:navatala_sparse_jacobi_davidson_correction_f32",
  "abi-r1:metal:navatala_sparse_jacobi_davidson_correction_f32",
  5,
  kAbiArgs_metal_navatala_sparse_jacobi_davidson_correction_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_jd_subspace_expand_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "t", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vNew", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "normNew", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_jd_subspace_expand_f32 = {
  1,
  "navatala_sparse_jd_subspace_expand_f32",
  "metal",
  "navatala_sparse_jd_subspace_expand_f32",
  "kernel:metal:navatala_sparse_jd_subspace_expand_f32",
  "abi-r1:metal:navatala_sparse_jd_subspace_expand_f32",
  "abi-r1:metal:navatala_sparse_jd_subspace_expand_f32",
  6,
  kAbiArgs_metal_navatala_sparse_jd_subspace_expand_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_subspace_orthogonalize_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "X", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_subspace_orthogonalize_f32 = {
  1,
  "navatala_sparse_subspace_orthogonalize_f32",
  "metal",
  "navatala_sparse_subspace_orthogonalize_f32",
  "kernel:metal:navatala_sparse_subspace_orthogonalize_f32",
  "abi-r1:metal:navatala_sparse_subspace_orthogonalize_f32",
  "abi-r1:metal:navatala_sparse_subspace_orthogonalize_f32",
  3,
  kAbiArgs_metal_navatala_sparse_subspace_orthogonalize_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_subspace_convergence_f32[] = {
  { "AX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdas", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "residNorms", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nConverged", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_subspace_convergence_f32 = {
  1,
  "navatala_sparse_subspace_convergence_f32",
  "metal",
  "navatala_sparse_subspace_convergence_f32",
  "kernel:metal:navatala_sparse_subspace_convergence_f32",
  "abi-r1:metal:navatala_sparse_subspace_convergence_f32",
  "abi-r1:metal:navatala_sparse_subspace_convergence_f32",
  8,
  kAbiArgs_metal_navatala_sparse_subspace_convergence_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_block_sp_m_v_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Y", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_block_sp_m_v_f32 = {
  1,
  "navatala_sparse_block_sp_m_v_f32",
  "metal",
  "navatala_sparse_block_sp_m_v_f32",
  "kernel:metal:navatala_sparse_block_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_block_sp_m_v_f32",
  "abi-r1:metal:navatala_sparse_block_sp_m_v_f32",
  7,
  kAbiArgs_metal_navatala_sparse_block_sp_m_v_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multi_vec_norm_partials_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multi_vec_norm_partials_f32 = {
  1,
  "navatala_sparse_multi_vec_norm_partials_f32",
  "metal",
  "navatala_sparse_multi_vec_norm_partials_f32",
  "kernel:metal:navatala_sparse_multi_vec_norm_partials_f32",
  "abi-r1:metal:navatala_sparse_multi_vec_norm_partials_f32",
  "abi-r1:metal:navatala_sparse_multi_vec_norm_partials_f32",
  4,
  kAbiArgs_metal_navatala_sparse_multi_vec_norm_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_multi_vec_scale_f32[] = {
  { "norms", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "X", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_multi_vec_scale_f32 = {
  1,
  "navatala_sparse_multi_vec_scale_f32",
  "metal",
  "navatala_sparse_multi_vec_scale_f32",
  "kernel:metal:navatala_sparse_multi_vec_scale_f32",
  "abi-r1:metal:navatala_sparse_multi_vec_scale_f32",
  "abi-r1:metal:navatala_sparse_multi_vec_scale_f32",
  4,
  kAbiArgs_metal_navatala_sparse_multi_vec_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_dense_sym_eig_f32[] = {
  { "M", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eigenvalues", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eigenvectors", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_dense_sym_eig_f32 = {
  1,
  "navatala_sparse_dense_sym_eig_f32",
  "metal",
  "navatala_sparse_dense_sym_eig_f32",
  "kernel:metal:navatala_sparse_dense_sym_eig_f32",
  "abi-r1:metal:navatala_sparse_dense_sym_eig_f32",
  "abi-r1:metal:navatala_sparse_dense_sym_eig_f32",
  4,
  kAbiArgs_metal_navatala_sparse_dense_sym_eig_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_convergence_relative_max_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_convergence_relative_max_f32 = {
  1,
  "navatala_sparse_convergence_relative_max_f32",
  "metal",
  "navatala_sparse_convergence_relative_max_f32",
  "kernel:metal:navatala_sparse_convergence_relative_max_f32",
  "abi-r1:metal:navatala_sparse_convergence_relative_max_f32",
  "abi-r1:metal:navatala_sparse_convergence_relative_max_f32",
  4,
  kAbiArgs_metal_navatala_sparse_convergence_relative_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_convergence_combined_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "initialNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "absTol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "relTol", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_convergence_combined_f32 = {
  1,
  "navatala_sparse_convergence_combined_f32",
  "metal",
  "navatala_sparse_convergence_combined_f32",
  "kernel:metal:navatala_sparse_convergence_combined_f32",
  "abi-r1:metal:navatala_sparse_convergence_combined_f32",
  "abi-r1:metal:navatala_sparse_convergence_combined_f32",
  5,
  kAbiArgs_metal_navatala_sparse_convergence_combined_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_convergence_divergence_check_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prevNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "consecutiveIncreases", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "diverged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newConsecutiveIncreases", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_convergence_divergence_check_f32 = {
  1,
  "navatala_sparse_convergence_divergence_check_f32",
  "metal",
  "navatala_sparse_convergence_divergence_check_f32",
  "kernel:metal:navatala_sparse_convergence_divergence_check_f32",
  "abi-r1:metal:navatala_sparse_convergence_divergence_check_f32",
  "abi-r1:metal:navatala_sparse_convergence_divergence_check_f32",
  5,
  kAbiArgs_metal_navatala_sparse_convergence_divergence_check_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_update_max_norm_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_update_max_norm_f32 = {
  1,
  "navatala_sparse_update_max_norm_f32",
  "metal",
  "navatala_sparse_update_max_norm_f32",
  "kernel:metal:navatala_sparse_update_max_norm_f32",
  "abi-r1:metal:navatala_sparse_update_max_norm_f32",
  "abi-r1:metal:navatala_sparse_update_max_norm_f32",
  2,
  kAbiArgs_metal_navatala_sparse_update_max_norm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_amg_level_metrics_f32[] = {
  { "levelNRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "levelNnz", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nLevels", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gridComplexity", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "operatorComplexity", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_amg_level_metrics_f32 = {
  1,
  "navatala_sparse_amg_level_metrics_f32",
  "metal",
  "navatala_sparse_amg_level_metrics_f32",
  "kernel:metal:navatala_sparse_amg_level_metrics_f32",
  "abi-r1:metal:navatala_sparse_amg_level_metrics_f32",
  "abi-r1:metal:navatala_sparse_amg_level_metrics_f32",
  5,
  kAbiArgs_metal_navatala_sparse_amg_level_metrics_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_amg_coarse_size_check_f32[] = {
  { "coarseNRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "denseLUThreshold", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "useDenseLU", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_amg_coarse_size_check_f32 = {
  1,
  "navatala_sparse_amg_coarse_size_check_f32",
  "metal",
  "navatala_sparse_amg_coarse_size_check_f32",
  "kernel:metal:navatala_sparse_amg_coarse_size_check_f32",
  "abi-r1:metal:navatala_sparse_amg_coarse_size_check_f32",
  "abi-r1:metal:navatala_sparse_amg_coarse_size_check_f32",
  3,
  kAbiArgs_metal_navatala_sparse_amg_coarse_size_check_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_amg_cycle_counter_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxCycles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cycleCount", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_amg_cycle_counter_f32 = {
  1,
  "navatala_sparse_amg_cycle_counter_f32",
  "metal",
  "navatala_sparse_amg_cycle_counter_f32",
  "kernel:metal:navatala_sparse_amg_cycle_counter_f32",
  "abi-r1:metal:navatala_sparse_amg_cycle_counter_f32",
  "abi-r1:metal:navatala_sparse_amg_cycle_counter_f32",
  5,
  kAbiArgs_metal_navatala_sparse_amg_cycle_counter_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_amg_validate_coarsening_f32[] = {
  { "fineNRows", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseNRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minRatio", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxRatio", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "acceptable", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_amg_validate_coarsening_f32 = {
  1,
  "navatala_sparse_amg_validate_coarsening_f32",
  "metal",
  "navatala_sparse_amg_validate_coarsening_f32",
  "kernel:metal:navatala_sparse_amg_validate_coarsening_f32",
  "abi-r1:metal:navatala_sparse_amg_validate_coarsening_f32",
  "abi-r1:metal:navatala_sparse_amg_validate_coarsening_f32",
  5,
  kAbiArgs_metal_navatala_sparse_amg_validate_coarsening_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_halo_pack_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sendMap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sendCount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sendBuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_halo_pack_f32 = {
  1,
  "navatala_sparse_halo_pack_f32",
  "metal",
  "navatala_sparse_halo_pack_f32",
  "kernel:metal:navatala_sparse_halo_pack_f32",
  "abi-r1:metal:navatala_sparse_halo_pack_f32",
  "abi-r1:metal:navatala_sparse_halo_pack_f32",
  4,
  kAbiArgs_metal_navatala_sparse_halo_pack_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_halo_unpack_f32[] = {
  { "recvBuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvMap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvCount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xHalo", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_halo_unpack_f32 = {
  1,
  "navatala_sparse_halo_unpack_f32",
  "metal",
  "navatala_sparse_halo_unpack_f32",
  "kernel:metal:navatala_sparse_halo_unpack_f32",
  "abi-r1:metal:navatala_sparse_halo_unpack_f32",
  "abi-r1:metal:navatala_sparse_halo_unpack_f32",
  4,
  kAbiArgs_metal_navatala_sparse_halo_unpack_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_halo_pack_b_s_r_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sendMap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sendCount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "blockDim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sendBuf", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_halo_pack_b_s_r_f32 = {
  1,
  "navatala_sparse_halo_pack_b_s_r_f32",
  "metal",
  "navatala_sparse_halo_pack_b_s_r_f32",
  "kernel:metal:navatala_sparse_halo_pack_b_s_r_f32",
  "abi-r1:metal:navatala_sparse_halo_pack_b_s_r_f32",
  "abi-r1:metal:navatala_sparse_halo_pack_b_s_r_f32",
  5,
  kAbiArgs_metal_navatala_sparse_halo_pack_b_s_r_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_partition_count_per_rank[] = {
  { "partitionVector", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRanks", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "countPerRank", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_partition_count_per_rank = {
  1,
  "navatala_sparse_partition_count_per_rank",
  "metal",
  "navatala_sparse_partition_count_per_rank",
  "kernel:metal:navatala_sparse_partition_count_per_rank",
  "abi-r1:metal:navatala_sparse_partition_count_per_rank",
  "abi-r1:metal:navatala_sparse_partition_count_per_rank",
  4,
  kAbiArgs_metal_navatala_sparse_partition_count_per_rank
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_partition_reorder_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "perm", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newRowPtr", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newColIdx", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newValues", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_partition_reorder_f32 = {
  1,
  "navatala_sparse_partition_reorder_f32",
  "metal",
  "navatala_sparse_partition_reorder_f32",
  "kernel:metal:navatala_sparse_partition_reorder_f32",
  "abi-r1:metal:navatala_sparse_partition_reorder_f32",
  "abi-r1:metal:navatala_sparse_partition_reorder_f32",
  8,
  kAbiArgs_metal_navatala_sparse_partition_reorder_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_global_to_local_index_map[] = {
  { "globalIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "localOffset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nLocal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nIndices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "localIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_global_to_local_index_map = {
  1,
  "navatala_sparse_global_to_local_index_map",
  "metal",
  "navatala_sparse_global_to_local_index_map",
  "kernel:metal:navatala_sparse_global_to_local_index_map",
  "abi-r1:metal:navatala_sparse_global_to_local_index_map",
  "abi-r1:metal:navatala_sparse_global_to_local_index_map",
  5,
  kAbiArgs_metal_navatala_sparse_global_to_local_index_map
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_distributed_sp_m_v_halo_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "xHalo", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yHalo", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_distributed_sp_m_v_halo_f32 = {
  1,
  "navatala_sparse_distributed_sp_m_v_halo_f32",
  "metal",
  "navatala_sparse_distributed_sp_m_v_halo_f32",
  "kernel:metal:navatala_sparse_distributed_sp_m_v_halo_f32",
  "abi-r1:metal:navatala_sparse_distributed_sp_m_v_halo_f32",
  "abi-r1:metal:navatala_sparse_distributed_sp_m_v_halo_f32",
  6,
  kAbiArgs_metal_navatala_sparse_distributed_sp_m_v_halo_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_merge_local_halo_result_f32[] = {
  { "yLocal", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "yHalo", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_merge_local_halo_result_f32 = {
  1,
  "navatala_sparse_merge_local_halo_result_f32",
  "metal",
  "navatala_sparse_merge_local_halo_result_f32",
  "kernel:metal:navatala_sparse_merge_local_halo_result_f32",
  "abi-r1:metal:navatala_sparse_merge_local_halo_result_f32",
  "abi-r1:metal:navatala_sparse_merge_local_halo_result_f32",
  4,
  kAbiArgs_metal_navatala_sparse_merge_local_halo_result_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f32[] = {
  { "nCells", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNnz", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f32 = {
  1,
  "navatala_sparse_ldu_to_csr_init_diagonal_count_f32",
  "metal",
  "navatala_sparse_ldu_to_csr_init_diagonal_count_f32",
  "kernel:metal:navatala_sparse_ldu_to_csr_init_diagonal_count_f32",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_init_diagonal_count_f32",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_init_diagonal_count_f32",
  2,
  kAbiArgs_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f64[] = {
  { "nCells", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNnz", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f64 = {
  1,
  "navatala_sparse_ldu_to_csr_init_diagonal_count_f64",
  "metal",
  "navatala_sparse_ldu_to_csr_init_diagonal_count_f64",
  "kernel:metal:navatala_sparse_ldu_to_csr_init_diagonal_count_f64",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_init_diagonal_count_f64",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_init_diagonal_count_f64",
  2,
  kAbiArgs_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ldu_to_csr_count_nnz_f32[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nInternalFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNnz", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ldu_to_csr_count_nnz_f32 = {
  1,
  "navatala_sparse_ldu_to_csr_count_nnz_f32",
  "metal",
  "navatala_sparse_ldu_to_csr_count_nnz_f32",
  "kernel:metal:navatala_sparse_ldu_to_csr_count_nnz_f32",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_count_nnz_f32",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_count_nnz_f32",
  5,
  kAbiArgs_metal_navatala_sparse_ldu_to_csr_count_nnz_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ldu_to_csr_count_nnz_f64[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nInternalFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowNnz", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ldu_to_csr_count_nnz_f64 = {
  1,
  "navatala_sparse_ldu_to_csr_count_nnz_f64",
  "metal",
  "navatala_sparse_ldu_to_csr_count_nnz_f64",
  "kernel:metal:navatala_sparse_ldu_to_csr_count_nnz_f64",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_count_nnz_f64",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_count_nnz_f64",
  5,
  kAbiArgs_metal_navatala_sparse_ldu_to_csr_count_nnz_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ldu_to_csr_fill_f32[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nInternalFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCells", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colInd", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "insertIdx", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ldu_to_csr_fill_f32 = {
  1,
  "navatala_sparse_ldu_to_csr_fill_f32",
  "metal",
  "navatala_sparse_ldu_to_csr_fill_f32",
  "kernel:metal:navatala_sparse_ldu_to_csr_fill_f32",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_fill_f32",
  "abi-r1:metal:navatala_sparse_ldu_to_csr_fill_f32",
  10,
  kAbiArgs_metal_navatala_sparse_ldu_to_csr_fill_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_ldu_diagonal_fill_f32[] = {
  { "diagonal", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colInd", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_ldu_diagonal_fill_f32 = {
  1,
  "navatala_sparse_ldu_diagonal_fill_f32",
  "metal",
  "navatala_sparse_ldu_diagonal_fill_f32",
  "kernel:metal:navatala_sparse_ldu_diagonal_fill_f32",
  "abi-r1:metal:navatala_sparse_ldu_diagonal_fill_f32",
  "abi-r1:metal:navatala_sparse_ldu_diagonal_fill_f32",
  5,
  kAbiArgs_metal_navatala_sparse_ldu_diagonal_fill_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_so_c_ldu_count_f32[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellFaceOffsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellFaceIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCells", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nnzPerRow", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_so_c_ldu_count_f32 = {
  1,
  "navatala_sparse_compute_so_c_ldu_count_f32",
  "metal",
  "navatala_sparse_compute_so_c_ldu_count_f32",
  "kernel:metal:navatala_sparse_compute_so_c_ldu_count_f32",
  "abi-r1:metal:navatala_sparse_compute_so_c_ldu_count_f32",
  "abi-r1:metal:navatala_sparse_compute_so_c_ldu_count_f32",
  8,
  kAbiArgs_metal_navatala_sparse_compute_so_c_ldu_count_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_so_c_ldu_fill_f32[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellFaceOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellFaceIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongRowPtr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCells", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "strongColIdx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_so_c_ldu_fill_f32 = {
  1,
  "navatala_sparse_compute_so_c_ldu_fill_f32",
  "metal",
  "navatala_sparse_compute_so_c_ldu_fill_f32",
  "kernel:metal:navatala_sparse_compute_so_c_ldu_fill_f32",
  "abi-r1:metal:navatala_sparse_compute_so_c_ldu_fill_f32",
  "abi-r1:metal:navatala_sparse_compute_so_c_ldu_fill_f32",
  10,
  kAbiArgs_metal_navatala_sparse_compute_so_c_ldu_fill_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_m_i_s_f32[] = {
  { "strongRowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongColIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomPriority", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nUndecided", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_m_i_s_f32 = {
  1,
  "navatala_sparse_compute_m_i_s_f32",
  "metal",
  "navatala_sparse_compute_m_i_s_f32",
  "kernel:metal:navatala_sparse_compute_m_i_s_f32",
  "abi-r1:metal:navatala_sparse_compute_m_i_s_f32",
  "abi-r1:metal:navatala_sparse_compute_m_i_s_f32",
  6,
  kAbiArgs_metal_navatala_sparse_compute_m_i_s_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_compute_m_i_s_f64[] = {
  { "strongRowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strongColIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomPriority", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aggregateId", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nUndecided", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_compute_m_i_s_f64 = {
  1,
  "navatala_sparse_compute_m_i_s_f64",
  "metal",
  "navatala_sparse_compute_m_i_s_f64",
  "kernel:metal:navatala_sparse_compute_m_i_s_f64",
  "abi-r1:metal:navatala_sparse_compute_m_i_s_f64",
  "abi-r1:metal:navatala_sparse_compute_m_i_s_f64",
  6,
  kAbiArgs_metal_navatala_sparse_compute_m_i_s_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_csr_sort_after_ldu_fill_f32[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_csr_sort_after_ldu_fill_f32 = {
  1,
  "navatala_sparse_csr_sort_after_ldu_fill_f32",
  "metal",
  "navatala_sparse_csr_sort_after_ldu_fill_f32",
  "kernel:metal:navatala_sparse_csr_sort_after_ldu_fill_f32",
  "abi-r1:metal:navatala_sparse_csr_sort_after_ldu_fill_f32",
  "abi-r1:metal:navatala_sparse_csr_sort_after_ldu_fill_f32",
  4,
  kAbiArgs_metal_navatala_sparse_csr_sort_after_ldu_fill_f32
};

bool tryGetKernelAbiManifest_metal_sparse(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_add_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_add_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_sub_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_sub_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_scale_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_hadamard_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_hadamard_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_abs_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_abs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_threshold_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_threshold_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_coo_transpose_f32") {
    out = &kAbiManifest_metal_navatala_sparse_coo_transpose_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_inner_product_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_inner_product_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_l2_distance_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_l2_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_cosine_distance_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_cosine_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_jaccard_distance_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_jaccard_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_jacobi_iter_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_jacobi_iter_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_gauss_seidel_iter_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_gauss_seidel_iter_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_cg_iter_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_cg_iter_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_bicgstab_iter_f32") {
    out = &kAbiManifest_metal_navatala_sparse_sparse_bicgstab_iter_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_residual_f32") {
    out = &kAbiManifest_metal_navatala_sparse_compute_residual_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_residual_norm_f32") {
    out = &kAbiManifest_metal_navatala_sparse_compute_residual_norm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_spmv_row_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_spmv_row_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chebyshev_smooth_f32") {
    out = &kAbiManifest_metal_navatala_sparse_chebyshev_smooth_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_find_min_edge_per_component_f32") {
    out = &kAbiManifest_metal_navatala_sparse_find_min_edge_per_component_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_merge_components_f32") {
    out = &kAbiManifest_metal_navatala_sparse_merge_components_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_merge_components_f64") {
    out = &kAbiManifest_metal_navatala_sparse_merge_components_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mst_boruvka_step_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mst_boruvka_step_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sp_m_v_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_sp_m_v_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sp_m_v_scaled_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_sp_m_v_scaled_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_residual_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_residual_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_axpy_f32") {
    out = &kAbiManifest_metal_navatala_sparse_axpy_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_xpay_f32") {
    out = &kAbiManifest_metal_navatala_sparse_xpay_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dot_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_dot_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_scal_f32") {
    out = &kAbiManifest_metal_navatala_sparse_scal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_nrm2_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_nrm2_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_zero_fill_f32") {
    out = &kAbiManifest_metal_navatala_sparse_zero_fill_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_zero_fill_u32") {
    out = &kAbiManifest_metal_navatala_sparse_zero_fill_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_zero_scalar_u32") {
    out = &kAbiManifest_metal_navatala_sparse_zero_scalar_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fill_neg_one_i32") {
    out = &kAbiManifest_metal_navatala_sparse_fill_neg_one_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_copy_f32") {
    out = &kAbiManifest_metal_navatala_sparse_copy_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bsr_sp_m_v_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bsr_sp_m_v_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_transpose_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_transpose_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_transpose_fill_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_transpose_fill_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_norm_l1_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_row_norm_l1_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_norm_linf_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_row_norm_linf_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_frobenius_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_frobenius_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_truncate_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_truncate_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_compact_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_compact_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sort_columns_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_sort_columns_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_permute_rows_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_permute_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_check_diag_dominance_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_check_diag_dominance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_nnz_histogram") {
    out = &kAbiManifest_metal_navatala_sparse_csr_row_nnz_histogram;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dense_l_u_f32") {
    out = &kAbiManifest_metal_navatala_sparse_dense_l_u_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dense_l_u_solve_f32") {
    out = &kAbiManifest_metal_navatala_sparse_dense_l_u_solve_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_replace_values_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_replace_values_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_to_coo_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_to_coo_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_coo_to_csr_row_count") {
    out = &kAbiManifest_metal_navatala_sparse_coo_to_csr_row_count;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_to_ell_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_to_ell_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ell_sp_m_v_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ell_sp_m_v_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_add_symbolic_row_nnz") {
    out = &kAbiManifest_metal_navatala_sparse_csr_add_symbolic_row_nnz;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_add_numeric_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_add_numeric_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_scale_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_symmetry_check_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_symmetry_check_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_nnz_per_row") {
    out = &kAbiManifest_metal_navatala_sparse_csr_nnz_per_row;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_max_nnz_per_row_partials") {
    out = &kAbiManifest_metal_navatala_sparse_csr_max_nnz_per_row_partials;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_extract_diag_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_extract_diag_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_permute_cols_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_permute_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_slice_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_row_slice_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_update_x_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_update_x_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_update_p_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_update_p_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_compute_alpha_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_compute_alpha_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_compute_beta_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_compute_beta_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_fused_update_dot_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_fused_update_dot_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_init_residual_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_init_residual_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_residual_norm_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cg_residual_norm_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_update_p_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_update_p_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_update_x_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_update_x_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_beta_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_compute_beta_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_omega_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_compute_omega_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_update_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_update_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_s_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_compute_s_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_alpha_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_compute_alpha_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_fused_update_all_f32") {
    out = &kAbiManifest_metal_navatala_sparse_bicgstab_fused_update_all_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_arnoldi_project_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_arnoldi_project_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_normalize_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_normalize_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_update_solution_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_update_solution_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_givens_rotation_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_givens_rotation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_arnoldi_batch_project_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_arnoldi_batch_project_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_hessenberg_dot_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_hessenberg_dot_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_restart_init_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_restart_init_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_back_solve_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_back_solve_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_store_basis_f32") {
    out = &kAbiManifest_metal_navatala_sparse_gmres_store_basis_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_shadow_project_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_shadow_project_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_update_x_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_update_x_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_g_space_update_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_g_space_update_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_compute_omega_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_compute_omega_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_build_shadow_space_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_build_shadow_space_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_m_sync_fused_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_m_sync_fused_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_compute_v_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_compute_v_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_store_d_r_column_f32") {
    out = &kAbiManifest_metal_navatala_sparse_idr_store_d_r_column_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pcg_update_x_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_pcg_update_x_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pcgf_update_p_f32") {
    out = &kAbiManifest_metal_navatala_sparse_pcgf_update_p_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pbicgstab_fused_s_f32") {
    out = &kAbiManifest_metal_navatala_sparse_pbicgstab_fused_s_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pbicgstab_triple_dot_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_pbicgstab_triple_dot_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fgmres_store_z_f32") {
    out = &kAbiManifest_metal_navatala_sparse_fgmres_store_z_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fgmres_update_solution_f32") {
    out = &kAbiManifest_metal_navatala_sparse_fgmres_update_solution_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pipe_c_g_fused_f32") {
    out = &kAbiManifest_metal_navatala_sparse_pipe_c_g_fused_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chronopoulos_c_g_scalars_f32") {
    out = &kAbiManifest_metal_navatala_sparse_chronopoulos_c_g_scalars_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fgmres_m_g_s_project_f32") {
    out = &kAbiManifest_metal_navatala_sparse_fgmres_m_g_s_project_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pbicgstab_fused_update_conv_f32") {
    out = &kAbiManifest_metal_navatala_sparse_pbicgstab_fused_update_conv_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jacobi_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_jacobi_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jacobi_l1_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_jacobi_l1_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_extract_diagonal_f32") {
    out = &kAbiManifest_metal_navatala_sparse_extract_diagonal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_block_jacobi_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_block_jacobi_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_jacobi_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cf_jacobi_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_gs_forward_f32") {
    out = &kAbiManifest_metal_navatala_sparse_multicolor_gs_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_gs_backward_f32") {
    out = &kAbiManifest_metal_navatala_sparse_multicolor_gs_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fixcolor_gs_forward_f32") {
    out = &kAbiManifest_metal_navatala_sparse_fixcolor_gs_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fixcolor_gs_backward_f32") {
    out = &kAbiManifest_metal_navatala_sparse_fixcolor_gs_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ilu_factor_color_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ilu_factor_color_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ilu_forward_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ilu_forward_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ilu_backward_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ilu_backward_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dilu_build_reciprocal_d_f32") {
    out = &kAbiManifest_metal_navatala_sparse_dilu_build_reciprocal_d_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dilu_apply_f32") {
    out = &kAbiManifest_metal_navatala_sparse_dilu_apply_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_dilu_forward_f32") {
    out = &kAbiManifest_metal_navatala_sparse_multicolor_dilu_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_dilu_backward_f32") {
    out = &kAbiManifest_metal_navatala_sparse_multicolor_dilu_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chebyshev_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_chebyshev_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chebyshev_coeff_update_f32") {
    out = &kAbiManifest_metal_navatala_sparse_chebyshev_coeff_update_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_polynomial_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_polynomial_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_kpz_polynomial_sweep_f32") {
    out = &kAbiManifest_metal_navatala_sparse_kpz_polynomial_sweep_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_kaczmarz_row_norm_sq_f32") {
    out = &kAbiManifest_metal_navatala_sparse_kaczmarz_row_norm_sq_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_of_connection_f32") {
    out = &kAbiManifest_metal_navatala_sparse_strength_of_connection_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size2_propose_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_size2_propose_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size2_confirm") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_size2_confirm;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_promote_unmatched_singletons") {
    out = &kAbiManifest_metal_navatala_sparse_promote_unmatched_singletons;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size4_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_size4_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size8_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_size8_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_parallel_greedy_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_parallel_greedy_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_assign_leader_compact_ids") {
    out = &kAbiManifest_metal_navatala_sparse_assign_leader_compact_ids;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_apply_compact_ids") {
    out = &kAbiManifest_metal_navatala_sparse_apply_compact_ids;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_prolongation_f32") {
    out = &kAbiManifest_metal_navatala_sparse_build_prolongation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_prolongation_terminator") {
    out = &kAbiManifest_metal_navatala_sparse_build_prolongation_terminator;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_symbolic_row_nnz") {
    out = &kAbiManifest_metal_navatala_sparse_spgemm_symbolic_row_nnz;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_numeric_f32") {
    out = &kAbiManifest_metal_navatala_sparse_spgemm_numeric_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_transpose") {
    out = &kAbiManifest_metal_navatala_sparse_csr_transpose;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_adaptive_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_adaptive_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_multi_pairwise_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_multi_pairwise_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_geometric_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_geometric_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_serial_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_serial_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_low_degree_first_propose_f32") {
    out = &kAbiManifest_metal_navatala_sparse_aggregate_low_degree_first_propose_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_assign_orphans_f32") {
    out = &kAbiManifest_metal_navatala_sparse_assign_orphans_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_numeric_hash_f32") {
    out = &kAbiManifest_metal_navatala_sparse_spgemm_numeric_hash_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_numeric_serial_f32") {
    out = &kAbiManifest_metal_navatala_sparse_spgemm_numeric_serial_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_smoothed_prolongation_f32") {
    out = &kAbiManifest_metal_navatala_sparse_build_smoothed_prolongation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_aggregate_sizes") {
    out = &kAbiManifest_metal_navatala_sparse_compute_aggregate_sizes;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_coarsening_ratio") {
    out = &kAbiManifest_metal_navatala_sparse_compute_coarsening_ratio;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_p_m_i_s") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_p_m_i_s;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_h_m_i_s") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_h_m_i_s;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_h_m_i_s_phase2_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_h_m_i_s_phase2_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_r_s") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_r_s;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance1_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_distance1_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance1_count_f64") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_distance1_count_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance1_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_distance1_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance2_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_distance2_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance2_count_f64") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_distance2_count_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance2_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_distance2_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_truncate_interpolation_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_truncate_interpolation_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_truncate_interpolation_f32") {
    out = &kAbiManifest_metal_navatala_sparse_truncate_interpolation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_aggressive_p_m_i_s") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_aggressive_p_m_i_s;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_aggressive_h_m_i_s") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_aggressive_h_m_i_s;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_c_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_c_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_c_l_j_p") {
    out = &kAbiManifest_metal_navatala_sparse_cf_split_c_l_j_p;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_multipass_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_multipass_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_multipass_count_f64") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_multipass_count_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_multipass_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_multipass_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_symmetric_f32") {
    out = &kAbiManifest_metal_navatala_sparse_strength_symmetric_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_algebraic_dist_f32") {
    out = &kAbiManifest_metal_navatala_sparse_strength_algebraic_dist_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_affinity_f32") {
    out = &kAbiManifest_metal_navatala_sparse_strength_affinity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_count_c_f_points") {
    out = &kAbiManifest_metal_navatala_sparse_count_c_f_points;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_grid_complexity") {
    out = &kAbiManifest_metal_navatala_sparse_compute_grid_complexity;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_operator_complexity") {
    out = &kAbiManifest_metal_navatala_sparse_compute_operator_complexity;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_ext_no_i_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_ext_no_i_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_ext_no_i_count_f64") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_ext_no_i_count_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_ext_no_i_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_ext_no_i_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_f_f_weights_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_f_f_weights_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_symbolic_row_nnz_f32") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_symbolic_row_nnz_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_symbolic_row_nnz_f64") {
    out = &kAbiManifest_metal_navatala_sparse_interpolate_symbolic_row_nnz_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_coarse_to_fine_map") {
    out = &kAbiManifest_metal_navatala_sparse_build_coarse_to_fine_map;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_fine_to_coarse_map_count") {
    out = &kAbiManifest_metal_navatala_sparse_build_fine_to_coarse_map_count;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_fine_to_coarse_map") {
    out = &kAbiManifest_metal_navatala_sparse_build_fine_to_coarse_map;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_initial_prolongation_f32") {
    out = &kAbiManifest_metal_navatala_sparse_em_initial_prolongation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_energy_gradient_f32") {
    out = &kAbiManifest_metal_navatala_sparse_em_energy_gradient_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_update_prolongation_f32") {
    out = &kAbiManifest_metal_navatala_sparse_em_update_prolongation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_normalize_p_f32") {
    out = &kAbiManifest_metal_navatala_sparse_em_normalize_p_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_energy_eval_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_em_energy_eval_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_parallel_greedy_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_parallel_greedy_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_min_max_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_min_max_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_min_max2_ring_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_min_max2_ring_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multi_hash_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_multi_hash_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_color_offsets") {
    out = &kAbiManifest_metal_navatala_sparse_build_color_offsets;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_reorder_c_s_r_by_color") {
    out = &kAbiManifest_metal_navatala_sparse_reorder_c_s_r_by_color;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_round_robin_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_round_robin_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bfs_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_bfs_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_locally_downwind_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_locally_downwind_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_greedy_recolor_f32") {
    out = &kAbiManifest_metal_navatala_sparse_greedy_recolor_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_greedy_recolor_f64") {
    out = &kAbiManifest_metal_navatala_sparse_greedy_recolor_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_validate_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_validate_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_color_histogram") {
    out = &kAbiManifest_metal_navatala_sparse_color_histogram;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_uniform_random_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_uniform_random_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_saturation_coloring") {
    out = &kAbiManifest_metal_navatala_sparse_saturation_coloring;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_reorder_vector_by_color_f32") {
    out = &kAbiManifest_metal_navatala_sparse_reorder_vector_by_color_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_diag_inv_sqrt_f32") {
    out = &kAbiManifest_metal_navatala_sparse_diag_inv_sqrt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_apply_diag_sym_scale_f32") {
    out = &kAbiManifest_metal_navatala_sparse_apply_diag_sym_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_binorm_row_scale_f32") {
    out = &kAbiManifest_metal_navatala_sparse_binorm_row_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_shifted_sp_m_v_f32") {
    out = &kAbiManifest_metal_navatala_sparse_shifted_sp_m_v_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_deflated_project_f32") {
    out = &kAbiManifest_metal_navatala_sparse_deflated_project_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_page_rank_iter_f32") {
    out = &kAbiManifest_metal_navatala_sparse_page_rank_iter_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_w_cycle_weight_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_w_cycle_weight_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_factor_f32") {
    out = &kAbiManifest_metal_navatala_sparse_convergence_factor_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_block_gram_schmidt_f32") {
    out = &kAbiManifest_metal_navatala_sparse_block_gram_schmidt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_small_dense_q_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_small_dense_q_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_lanczos_step_f32") {
    out = &kAbiManifest_metal_navatala_sparse_lanczos_step_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_arnoldi_step_f32") {
    out = &kAbiManifest_metal_navatala_sparse_arnoldi_step_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_lobpcg_rayleigh_ritz_f32") {
    out = &kAbiManifest_metal_navatala_sparse_lobpcg_rayleigh_ritz_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_lobpcg_residual_f32") {
    out = &kAbiManifest_metal_navatala_sparse_lobpcg_residual_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_power_iteration_normalize_f32") {
    out = &kAbiManifest_metal_navatala_sparse_power_iteration_normalize_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_rayleigh_quotient_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_rayleigh_quotient_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jacobi_davidson_correction_f32") {
    out = &kAbiManifest_metal_navatala_sparse_jacobi_davidson_correction_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jd_subspace_expand_f32") {
    out = &kAbiManifest_metal_navatala_sparse_jd_subspace_expand_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_subspace_orthogonalize_f32") {
    out = &kAbiManifest_metal_navatala_sparse_subspace_orthogonalize_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_subspace_convergence_f32") {
    out = &kAbiManifest_metal_navatala_sparse_subspace_convergence_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_block_sp_m_v_f32") {
    out = &kAbiManifest_metal_navatala_sparse_block_sp_m_v_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multi_vec_norm_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_multi_vec_norm_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multi_vec_scale_f32") {
    out = &kAbiManifest_metal_navatala_sparse_multi_vec_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dense_sym_eig_f32") {
    out = &kAbiManifest_metal_navatala_sparse_dense_sym_eig_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_relative_max_f32") {
    out = &kAbiManifest_metal_navatala_sparse_convergence_relative_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_combined_f32") {
    out = &kAbiManifest_metal_navatala_sparse_convergence_combined_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_divergence_check_f32") {
    out = &kAbiManifest_metal_navatala_sparse_convergence_divergence_check_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_update_max_norm_f32") {
    out = &kAbiManifest_metal_navatala_sparse_update_max_norm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_level_metrics_f32") {
    out = &kAbiManifest_metal_navatala_sparse_amg_level_metrics_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_coarse_size_check_f32") {
    out = &kAbiManifest_metal_navatala_sparse_amg_coarse_size_check_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_cycle_counter_f32") {
    out = &kAbiManifest_metal_navatala_sparse_amg_cycle_counter_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_validate_coarsening_f32") {
    out = &kAbiManifest_metal_navatala_sparse_amg_validate_coarsening_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_halo_pack_f32") {
    out = &kAbiManifest_metal_navatala_sparse_halo_pack_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_halo_unpack_f32") {
    out = &kAbiManifest_metal_navatala_sparse_halo_unpack_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_halo_pack_b_s_r_f32") {
    out = &kAbiManifest_metal_navatala_sparse_halo_pack_b_s_r_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_partition_count_per_rank") {
    out = &kAbiManifest_metal_navatala_sparse_partition_count_per_rank;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_partition_reorder_f32") {
    out = &kAbiManifest_metal_navatala_sparse_partition_reorder_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_global_to_local_index_map") {
    out = &kAbiManifest_metal_navatala_sparse_global_to_local_index_map;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_distributed_sp_m_v_halo_f32") {
    out = &kAbiManifest_metal_navatala_sparse_distributed_sp_m_v_halo_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_merge_local_halo_result_f32") {
    out = &kAbiManifest_metal_navatala_sparse_merge_local_halo_result_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_init_diagonal_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_init_diagonal_count_f64") {
    out = &kAbiManifest_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_count_nnz_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ldu_to_csr_count_nnz_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_count_nnz_f64") {
    out = &kAbiManifest_metal_navatala_sparse_ldu_to_csr_count_nnz_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_fill_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ldu_to_csr_fill_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_diagonal_fill_f32") {
    out = &kAbiManifest_metal_navatala_sparse_ldu_diagonal_fill_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_so_c_ldu_count_f32") {
    out = &kAbiManifest_metal_navatala_sparse_compute_so_c_ldu_count_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_so_c_ldu_fill_f32") {
    out = &kAbiManifest_metal_navatala_sparse_compute_so_c_ldu_fill_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_m_i_s_f32") {
    out = &kAbiManifest_metal_navatala_sparse_compute_m_i_s_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_m_i_s_f64") {
    out = &kAbiManifest_metal_navatala_sparse_compute_m_i_s_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sort_after_ldu_fill_f32") {
    out = &kAbiManifest_metal_navatala_sparse_csr_sort_after_ldu_fill_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_sparse(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_add_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_sub_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_scale_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_hadamard_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_hadamard_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_hadamard_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_abs_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_threshold_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_threshold_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_threshold_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_coo_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_coo_transpose_f32";
    std::string_view sv(k_metal_navatala_sparse_coo_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_inner_product_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_l2_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_l2_distance_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_l2_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_cosine_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_cosine_distance_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_cosine_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_jaccard_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_jaccard_distance_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_jaccard_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_jacobi_iter_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_jacobi_iter_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_jacobi_iter_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_gauss_seidel_iter_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_gauss_seidel_iter_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_gauss_seidel_iter_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_cg_iter_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_cg_iter_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_cg_iter_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_sparse_bicgstab_iter_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_sparse_bicgstab_iter_f32";
    std::string_view sv(k_metal_navatala_sparse_sparse_bicgstab_iter_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_residual_f32";
    std::string_view sv(k_metal_navatala_sparse_compute_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_residual_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_residual_norm_f32";
    std::string_view sv(k_metal_navatala_sparse_compute_residual_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_spmv_row_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_spmv_row_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_spmv_row_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chebyshev_smooth_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_chebyshev_smooth_f32";
    std::string_view sv(k_metal_navatala_sparse_chebyshev_smooth_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_find_min_edge_per_component_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_find_min_edge_per_component_f32";
    std::string_view sv(k_metal_navatala_sparse_find_min_edge_per_component_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_merge_components_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_merge_components_f32";
    std::string_view sv(k_metal_navatala_sparse_merge_components_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_merge_components_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_merge_components_f64";
    std::string_view sv(k_metal_navatala_sparse_merge_components_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mst_boruvka_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mst_boruvka_step_f32";
    std::string_view sv(k_metal_navatala_sparse_mst_boruvka_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sp_m_v_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_sp_m_v_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_sp_m_v_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sp_m_v_scaled_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_sp_m_v_scaled_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_sp_m_v_scaled_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_residual_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_axpy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_axpy_f32";
    std::string_view sv(k_metal_navatala_sparse_axpy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_xpay_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_xpay_f32";
    std::string_view sv(k_metal_navatala_sparse_xpay_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dot_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_dot_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_dot_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_scal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_scal_f32";
    std::string_view sv(k_metal_navatala_sparse_scal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_nrm2_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_nrm2_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_nrm2_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_zero_fill_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_zero_fill_f32";
    std::string_view sv(k_metal_navatala_sparse_zero_fill_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_zero_fill_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_zero_fill_u32";
    std::string_view sv(k_metal_navatala_sparse_zero_fill_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_zero_scalar_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_zero_scalar_u32";
    std::string_view sv(k_metal_navatala_sparse_zero_scalar_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fill_neg_one_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_fill_neg_one_i32";
    std::string_view sv(k_metal_navatala_sparse_fill_neg_one_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_copy_f32";
    std::string_view sv(k_metal_navatala_sparse_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bsr_sp_m_v_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bsr_sp_m_v_f32";
    std::string_view sv(k_metal_navatala_sparse_bsr_sp_m_v_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_transpose_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_transpose_count_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_transpose_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_transpose_fill_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_transpose_fill_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_transpose_fill_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_norm_l1_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_row_norm_l1_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_row_norm_l1_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_norm_linf_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_row_norm_linf_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_row_norm_linf_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_frobenius_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_frobenius_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_frobenius_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_truncate_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_truncate_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_truncate_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_compact_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_compact_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_compact_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sort_columns_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_sort_columns_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_sort_columns_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_permute_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_permute_rows_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_permute_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_check_diag_dominance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_check_diag_dominance_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_check_diag_dominance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_nnz_histogram") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_row_nnz_histogram";
    std::string_view sv(k_metal_navatala_sparse_csr_row_nnz_histogram);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dense_l_u_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_dense_l_u_f32";
    std::string_view sv(k_metal_navatala_sparse_dense_l_u_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dense_l_u_solve_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_dense_l_u_solve_f32";
    std::string_view sv(k_metal_navatala_sparse_dense_l_u_solve_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_replace_values_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_replace_values_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_replace_values_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_to_coo_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_to_coo_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_to_coo_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_coo_to_csr_row_count") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_coo_to_csr_row_count";
    std::string_view sv(k_metal_navatala_sparse_coo_to_csr_row_count);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_to_ell_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_to_ell_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_to_ell_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ell_sp_m_v_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ell_sp_m_v_f32";
    std::string_view sv(k_metal_navatala_sparse_ell_sp_m_v_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_add_symbolic_row_nnz") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_add_symbolic_row_nnz";
    std::string_view sv(k_metal_navatala_sparse_csr_add_symbolic_row_nnz);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_add_numeric_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_add_numeric_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_add_numeric_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_scale_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_symmetry_check_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_symmetry_check_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_symmetry_check_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_nnz_per_row") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_nnz_per_row";
    std::string_view sv(k_metal_navatala_sparse_csr_nnz_per_row);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_max_nnz_per_row_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_max_nnz_per_row_partials";
    std::string_view sv(k_metal_navatala_sparse_csr_max_nnz_per_row_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_extract_diag_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_extract_diag_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_extract_diag_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_permute_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_permute_cols_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_permute_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_row_slice_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_row_slice_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_row_slice_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_update_x_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_update_x_r_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_update_x_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_update_p_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_update_p_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_update_p_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_compute_alpha_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_compute_alpha_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_compute_alpha_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_compute_beta_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_compute_beta_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_compute_beta_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_fused_update_dot_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_fused_update_dot_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_fused_update_dot_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_init_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_init_residual_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_init_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cg_residual_norm_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cg_residual_norm_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_cg_residual_norm_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_update_p_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_update_p_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_update_p_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_update_x_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_update_x_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_update_x_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_beta_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_compute_beta_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_compute_beta_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_omega_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_compute_omega_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_compute_omega_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_update_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_update_r_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_update_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_s_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_compute_s_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_compute_s_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_compute_alpha_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_compute_alpha_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_compute_alpha_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bicgstab_fused_update_all_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bicgstab_fused_update_all_f32";
    std::string_view sv(k_metal_navatala_sparse_bicgstab_fused_update_all_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_arnoldi_project_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_arnoldi_project_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_arnoldi_project_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_normalize_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_normalize_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_normalize_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_update_solution_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_update_solution_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_update_solution_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_givens_rotation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_givens_rotation_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_givens_rotation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_arnoldi_batch_project_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_arnoldi_batch_project_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_arnoldi_batch_project_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_hessenberg_dot_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_hessenberg_dot_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_hessenberg_dot_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_restart_init_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_restart_init_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_restart_init_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_back_solve_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_back_solve_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_back_solve_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_gmres_store_basis_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_gmres_store_basis_f32";
    std::string_view sv(k_metal_navatala_sparse_gmres_store_basis_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_shadow_project_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_shadow_project_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_shadow_project_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_update_x_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_update_x_r_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_update_x_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_g_space_update_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_g_space_update_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_g_space_update_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_compute_omega_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_compute_omega_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_compute_omega_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_build_shadow_space_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_build_shadow_space_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_build_shadow_space_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_m_sync_fused_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_m_sync_fused_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_m_sync_fused_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_compute_v_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_compute_v_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_compute_v_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_idr_store_d_r_column_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_idr_store_d_r_column_f32";
    std::string_view sv(k_metal_navatala_sparse_idr_store_d_r_column_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pcg_update_x_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_pcg_update_x_r_f32";
    std::string_view sv(k_metal_navatala_sparse_pcg_update_x_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pcgf_update_p_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_pcgf_update_p_f32";
    std::string_view sv(k_metal_navatala_sparse_pcgf_update_p_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pbicgstab_fused_s_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_pbicgstab_fused_s_f32";
    std::string_view sv(k_metal_navatala_sparse_pbicgstab_fused_s_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pbicgstab_triple_dot_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_pbicgstab_triple_dot_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_pbicgstab_triple_dot_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fgmres_store_z_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_fgmres_store_z_f32";
    std::string_view sv(k_metal_navatala_sparse_fgmres_store_z_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fgmres_update_solution_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_fgmres_update_solution_f32";
    std::string_view sv(k_metal_navatala_sparse_fgmres_update_solution_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pipe_c_g_fused_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_pipe_c_g_fused_f32";
    std::string_view sv(k_metal_navatala_sparse_pipe_c_g_fused_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chronopoulos_c_g_scalars_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_chronopoulos_c_g_scalars_f32";
    std::string_view sv(k_metal_navatala_sparse_chronopoulos_c_g_scalars_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fgmres_m_g_s_project_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_fgmres_m_g_s_project_f32";
    std::string_view sv(k_metal_navatala_sparse_fgmres_m_g_s_project_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_pbicgstab_fused_update_conv_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_pbicgstab_fused_update_conv_f32";
    std::string_view sv(k_metal_navatala_sparse_pbicgstab_fused_update_conv_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jacobi_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_jacobi_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_jacobi_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jacobi_l1_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_jacobi_l1_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_jacobi_l1_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_extract_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_extract_diagonal_f32";
    std::string_view sv(k_metal_navatala_sparse_extract_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_block_jacobi_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_block_jacobi_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_block_jacobi_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_jacobi_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_jacobi_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_cf_jacobi_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_gs_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multicolor_gs_forward_f32";
    std::string_view sv(k_metal_navatala_sparse_multicolor_gs_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_gs_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multicolor_gs_backward_f32";
    std::string_view sv(k_metal_navatala_sparse_multicolor_gs_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fixcolor_gs_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_fixcolor_gs_forward_f32";
    std::string_view sv(k_metal_navatala_sparse_fixcolor_gs_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_fixcolor_gs_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_fixcolor_gs_backward_f32";
    std::string_view sv(k_metal_navatala_sparse_fixcolor_gs_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ilu_factor_color_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ilu_factor_color_f32";
    std::string_view sv(k_metal_navatala_sparse_ilu_factor_color_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ilu_forward_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ilu_forward_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_ilu_forward_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ilu_backward_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ilu_backward_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_ilu_backward_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dilu_build_reciprocal_d_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_dilu_build_reciprocal_d_f32";
    std::string_view sv(k_metal_navatala_sparse_dilu_build_reciprocal_d_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dilu_apply_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_dilu_apply_f32";
    std::string_view sv(k_metal_navatala_sparse_dilu_apply_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_dilu_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multicolor_dilu_forward_f32";
    std::string_view sv(k_metal_navatala_sparse_multicolor_dilu_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multicolor_dilu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multicolor_dilu_backward_f32";
    std::string_view sv(k_metal_navatala_sparse_multicolor_dilu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chebyshev_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_chebyshev_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_chebyshev_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_chebyshev_coeff_update_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_chebyshev_coeff_update_f32";
    std::string_view sv(k_metal_navatala_sparse_chebyshev_coeff_update_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_polynomial_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_polynomial_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_polynomial_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_kpz_polynomial_sweep_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_kpz_polynomial_sweep_f32";
    std::string_view sv(k_metal_navatala_sparse_kpz_polynomial_sweep_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_kaczmarz_row_norm_sq_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_kaczmarz_row_norm_sq_f32";
    std::string_view sv(k_metal_navatala_sparse_kaczmarz_row_norm_sq_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_of_connection_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_strength_of_connection_f32";
    std::string_view sv(k_metal_navatala_sparse_strength_of_connection_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size2_propose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_size2_propose_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_size2_propose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size2_confirm") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_size2_confirm";
    std::string_view sv(k_metal_navatala_sparse_aggregate_size2_confirm);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_promote_unmatched_singletons") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_promote_unmatched_singletons";
    std::string_view sv(k_metal_navatala_sparse_promote_unmatched_singletons);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size4_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_size4_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_size4_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_size8_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_size8_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_size8_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_parallel_greedy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_parallel_greedy_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_parallel_greedy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_assign_leader_compact_ids") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_assign_leader_compact_ids";
    std::string_view sv(k_metal_navatala_sparse_assign_leader_compact_ids);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_apply_compact_ids") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_apply_compact_ids";
    std::string_view sv(k_metal_navatala_sparse_apply_compact_ids);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_prolongation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_prolongation_f32";
    std::string_view sv(k_metal_navatala_sparse_build_prolongation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_prolongation_terminator") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_prolongation_terminator";
    std::string_view sv(k_metal_navatala_sparse_build_prolongation_terminator);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_symbolic_row_nnz") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_spgemm_symbolic_row_nnz";
    std::string_view sv(k_metal_navatala_sparse_spgemm_symbolic_row_nnz);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_numeric_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_spgemm_numeric_f32";
    std::string_view sv(k_metal_navatala_sparse_spgemm_numeric_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_transpose") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_transpose";
    std::string_view sv(k_metal_navatala_sparse_csr_transpose);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_adaptive_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_adaptive_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_adaptive_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_multi_pairwise_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_multi_pairwise_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_multi_pairwise_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_geometric_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_geometric_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_geometric_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_serial_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_serial_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_serial_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_aggregate_low_degree_first_propose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_aggregate_low_degree_first_propose_f32";
    std::string_view sv(k_metal_navatala_sparse_aggregate_low_degree_first_propose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_assign_orphans_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_assign_orphans_f32";
    std::string_view sv(k_metal_navatala_sparse_assign_orphans_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_numeric_hash_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_spgemm_numeric_hash_f32";
    std::string_view sv(k_metal_navatala_sparse_spgemm_numeric_hash_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_spgemm_numeric_serial_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_spgemm_numeric_serial_f32";
    std::string_view sv(k_metal_navatala_sparse_spgemm_numeric_serial_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_smoothed_prolongation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_smoothed_prolongation_f32";
    std::string_view sv(k_metal_navatala_sparse_build_smoothed_prolongation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_aggregate_sizes") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_aggregate_sizes";
    std::string_view sv(k_metal_navatala_sparse_compute_aggregate_sizes);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_coarsening_ratio") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_coarsening_ratio";
    std::string_view sv(k_metal_navatala_sparse_compute_coarsening_ratio);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_p_m_i_s") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_p_m_i_s";
    std::string_view sv(k_metal_navatala_sparse_cf_split_p_m_i_s);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_h_m_i_s") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_h_m_i_s";
    std::string_view sv(k_metal_navatala_sparse_cf_split_h_m_i_s);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_h_m_i_s_phase2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_h_m_i_s_phase2_f32";
    std::string_view sv(k_metal_navatala_sparse_cf_split_h_m_i_s_phase2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_r_s") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_r_s";
    std::string_view sv(k_metal_navatala_sparse_cf_split_r_s);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance1_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_distance1_count_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_distance1_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance1_count_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_distance1_count_f64";
    std::string_view sv(k_metal_navatala_sparse_interpolate_distance1_count_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance1_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_distance1_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_distance1_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance2_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_distance2_count_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_distance2_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance2_count_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_distance2_count_f64";
    std::string_view sv(k_metal_navatala_sparse_interpolate_distance2_count_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_distance2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_distance2_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_distance2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_truncate_interpolation_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_truncate_interpolation_count_f32";
    std::string_view sv(k_metal_navatala_sparse_truncate_interpolation_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_truncate_interpolation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_truncate_interpolation_f32";
    std::string_view sv(k_metal_navatala_sparse_truncate_interpolation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_aggressive_p_m_i_s") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_aggressive_p_m_i_s";
    std::string_view sv(k_metal_navatala_sparse_cf_split_aggressive_p_m_i_s);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_aggressive_h_m_i_s") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_aggressive_h_m_i_s";
    std::string_view sv(k_metal_navatala_sparse_cf_split_aggressive_h_m_i_s);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_c_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_c_r_f32";
    std::string_view sv(k_metal_navatala_sparse_cf_split_c_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_cf_split_c_l_j_p") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_cf_split_c_l_j_p";
    std::string_view sv(k_metal_navatala_sparse_cf_split_c_l_j_p);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_multipass_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_multipass_count_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_multipass_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_multipass_count_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_multipass_count_f64";
    std::string_view sv(k_metal_navatala_sparse_interpolate_multipass_count_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_multipass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_multipass_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_multipass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_symmetric_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_strength_symmetric_f32";
    std::string_view sv(k_metal_navatala_sparse_strength_symmetric_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_algebraic_dist_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_strength_algebraic_dist_f32";
    std::string_view sv(k_metal_navatala_sparse_strength_algebraic_dist_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_strength_affinity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_strength_affinity_f32";
    std::string_view sv(k_metal_navatala_sparse_strength_affinity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_count_c_f_points") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_count_c_f_points";
    std::string_view sv(k_metal_navatala_sparse_count_c_f_points);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_grid_complexity") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_grid_complexity";
    std::string_view sv(k_metal_navatala_sparse_compute_grid_complexity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_operator_complexity") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_operator_complexity";
    std::string_view sv(k_metal_navatala_sparse_compute_operator_complexity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_ext_no_i_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_ext_no_i_count_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_ext_no_i_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_ext_no_i_count_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_ext_no_i_count_f64";
    std::string_view sv(k_metal_navatala_sparse_interpolate_ext_no_i_count_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_ext_no_i_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_ext_no_i_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_ext_no_i_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_f_f_weights_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_f_f_weights_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_f_f_weights_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_symbolic_row_nnz_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_symbolic_row_nnz_f32";
    std::string_view sv(k_metal_navatala_sparse_interpolate_symbolic_row_nnz_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_interpolate_symbolic_row_nnz_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_interpolate_symbolic_row_nnz_f64";
    std::string_view sv(k_metal_navatala_sparse_interpolate_symbolic_row_nnz_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_coarse_to_fine_map") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_coarse_to_fine_map";
    std::string_view sv(k_metal_navatala_sparse_build_coarse_to_fine_map);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_fine_to_coarse_map_count") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_fine_to_coarse_map_count";
    std::string_view sv(k_metal_navatala_sparse_build_fine_to_coarse_map_count);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_fine_to_coarse_map") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_fine_to_coarse_map";
    std::string_view sv(k_metal_navatala_sparse_build_fine_to_coarse_map);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_initial_prolongation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_em_initial_prolongation_f32";
    std::string_view sv(k_metal_navatala_sparse_em_initial_prolongation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_energy_gradient_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_em_energy_gradient_f32";
    std::string_view sv(k_metal_navatala_sparse_em_energy_gradient_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_update_prolongation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_em_update_prolongation_f32";
    std::string_view sv(k_metal_navatala_sparse_em_update_prolongation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_normalize_p_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_em_normalize_p_f32";
    std::string_view sv(k_metal_navatala_sparse_em_normalize_p_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_em_energy_eval_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_em_energy_eval_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_em_energy_eval_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_parallel_greedy_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_parallel_greedy_coloring";
    std::string_view sv(k_metal_navatala_sparse_parallel_greedy_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_min_max_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_min_max_coloring";
    std::string_view sv(k_metal_navatala_sparse_min_max_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_min_max2_ring_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_min_max2_ring_coloring";
    std::string_view sv(k_metal_navatala_sparse_min_max2_ring_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multi_hash_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multi_hash_coloring";
    std::string_view sv(k_metal_navatala_sparse_multi_hash_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_build_color_offsets") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_build_color_offsets";
    std::string_view sv(k_metal_navatala_sparse_build_color_offsets);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_reorder_c_s_r_by_color") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_reorder_c_s_r_by_color";
    std::string_view sv(k_metal_navatala_sparse_reorder_c_s_r_by_color);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_round_robin_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_round_robin_coloring";
    std::string_view sv(k_metal_navatala_sparse_round_robin_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_bfs_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_bfs_coloring";
    std::string_view sv(k_metal_navatala_sparse_bfs_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_locally_downwind_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_locally_downwind_coloring";
    std::string_view sv(k_metal_navatala_sparse_locally_downwind_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_greedy_recolor_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_greedy_recolor_f32";
    std::string_view sv(k_metal_navatala_sparse_greedy_recolor_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_greedy_recolor_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_greedy_recolor_f64";
    std::string_view sv(k_metal_navatala_sparse_greedy_recolor_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_validate_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_validate_coloring";
    std::string_view sv(k_metal_navatala_sparse_validate_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_color_histogram") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_color_histogram";
    std::string_view sv(k_metal_navatala_sparse_color_histogram);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_uniform_random_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_uniform_random_coloring";
    std::string_view sv(k_metal_navatala_sparse_uniform_random_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_saturation_coloring") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_saturation_coloring";
    std::string_view sv(k_metal_navatala_sparse_saturation_coloring);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_reorder_vector_by_color_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_reorder_vector_by_color_f32";
    std::string_view sv(k_metal_navatala_sparse_reorder_vector_by_color_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_diag_inv_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_diag_inv_sqrt_f32";
    std::string_view sv(k_metal_navatala_sparse_diag_inv_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_apply_diag_sym_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_apply_diag_sym_scale_f32";
    std::string_view sv(k_metal_navatala_sparse_apply_diag_sym_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_binorm_row_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_binorm_row_scale_f32";
    std::string_view sv(k_metal_navatala_sparse_binorm_row_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_shifted_sp_m_v_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_shifted_sp_m_v_f32";
    std::string_view sv(k_metal_navatala_sparse_shifted_sp_m_v_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_deflated_project_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_deflated_project_f32";
    std::string_view sv(k_metal_navatala_sparse_deflated_project_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_page_rank_iter_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_page_rank_iter_f32";
    std::string_view sv(k_metal_navatala_sparse_page_rank_iter_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_w_cycle_weight_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_w_cycle_weight_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_w_cycle_weight_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_factor_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_convergence_factor_f32";
    std::string_view sv(k_metal_navatala_sparse_convergence_factor_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_block_gram_schmidt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_block_gram_schmidt_f32";
    std::string_view sv(k_metal_navatala_sparse_block_gram_schmidt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_small_dense_q_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_small_dense_q_r_f32";
    std::string_view sv(k_metal_navatala_sparse_small_dense_q_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_lanczos_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_lanczos_step_f32";
    std::string_view sv(k_metal_navatala_sparse_lanczos_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_arnoldi_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_arnoldi_step_f32";
    std::string_view sv(k_metal_navatala_sparse_arnoldi_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_lobpcg_rayleigh_ritz_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_lobpcg_rayleigh_ritz_f32";
    std::string_view sv(k_metal_navatala_sparse_lobpcg_rayleigh_ritz_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_lobpcg_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_lobpcg_residual_f32";
    std::string_view sv(k_metal_navatala_sparse_lobpcg_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_power_iteration_normalize_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_power_iteration_normalize_f32";
    std::string_view sv(k_metal_navatala_sparse_power_iteration_normalize_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_rayleigh_quotient_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_rayleigh_quotient_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_rayleigh_quotient_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jacobi_davidson_correction_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_jacobi_davidson_correction_f32";
    std::string_view sv(k_metal_navatala_sparse_jacobi_davidson_correction_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_jd_subspace_expand_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_jd_subspace_expand_f32";
    std::string_view sv(k_metal_navatala_sparse_jd_subspace_expand_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_subspace_orthogonalize_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_subspace_orthogonalize_f32";
    std::string_view sv(k_metal_navatala_sparse_subspace_orthogonalize_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_subspace_convergence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_subspace_convergence_f32";
    std::string_view sv(k_metal_navatala_sparse_subspace_convergence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_block_sp_m_v_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_block_sp_m_v_f32";
    std::string_view sv(k_metal_navatala_sparse_block_sp_m_v_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multi_vec_norm_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multi_vec_norm_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_multi_vec_norm_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_multi_vec_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_multi_vec_scale_f32";
    std::string_view sv(k_metal_navatala_sparse_multi_vec_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_dense_sym_eig_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_dense_sym_eig_f32";
    std::string_view sv(k_metal_navatala_sparse_dense_sym_eig_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_relative_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_convergence_relative_max_f32";
    std::string_view sv(k_metal_navatala_sparse_convergence_relative_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_combined_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_convergence_combined_f32";
    std::string_view sv(k_metal_navatala_sparse_convergence_combined_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_convergence_divergence_check_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_convergence_divergence_check_f32";
    std::string_view sv(k_metal_navatala_sparse_convergence_divergence_check_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_update_max_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_update_max_norm_f32";
    std::string_view sv(k_metal_navatala_sparse_update_max_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_level_metrics_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_amg_level_metrics_f32";
    std::string_view sv(k_metal_navatala_sparse_amg_level_metrics_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_coarse_size_check_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_amg_coarse_size_check_f32";
    std::string_view sv(k_metal_navatala_sparse_amg_coarse_size_check_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_cycle_counter_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_amg_cycle_counter_f32";
    std::string_view sv(k_metal_navatala_sparse_amg_cycle_counter_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_amg_validate_coarsening_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_amg_validate_coarsening_f32";
    std::string_view sv(k_metal_navatala_sparse_amg_validate_coarsening_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_halo_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_halo_pack_f32";
    std::string_view sv(k_metal_navatala_sparse_halo_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_halo_unpack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_halo_unpack_f32";
    std::string_view sv(k_metal_navatala_sparse_halo_unpack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_halo_pack_b_s_r_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_halo_pack_b_s_r_f32";
    std::string_view sv(k_metal_navatala_sparse_halo_pack_b_s_r_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_partition_count_per_rank") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_partition_count_per_rank";
    std::string_view sv(k_metal_navatala_sparse_partition_count_per_rank);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_partition_reorder_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_partition_reorder_f32";
    std::string_view sv(k_metal_navatala_sparse_partition_reorder_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_global_to_local_index_map") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_global_to_local_index_map";
    std::string_view sv(k_metal_navatala_sparse_global_to_local_index_map);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_distributed_sp_m_v_halo_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_distributed_sp_m_v_halo_f32";
    std::string_view sv(k_metal_navatala_sparse_distributed_sp_m_v_halo_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_merge_local_halo_result_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_merge_local_halo_result_f32";
    std::string_view sv(k_metal_navatala_sparse_merge_local_halo_result_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_init_diagonal_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ldu_to_csr_init_diagonal_count_f32";
    std::string_view sv(k_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_init_diagonal_count_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ldu_to_csr_init_diagonal_count_f64";
    std::string_view sv(k_metal_navatala_sparse_ldu_to_csr_init_diagonal_count_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_count_nnz_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ldu_to_csr_count_nnz_f32";
    std::string_view sv(k_metal_navatala_sparse_ldu_to_csr_count_nnz_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_count_nnz_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ldu_to_csr_count_nnz_f64";
    std::string_view sv(k_metal_navatala_sparse_ldu_to_csr_count_nnz_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_to_csr_fill_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ldu_to_csr_fill_f32";
    std::string_view sv(k_metal_navatala_sparse_ldu_to_csr_fill_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_ldu_diagonal_fill_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_ldu_diagonal_fill_f32";
    std::string_view sv(k_metal_navatala_sparse_ldu_diagonal_fill_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_so_c_ldu_count_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_so_c_ldu_count_f32";
    std::string_view sv(k_metal_navatala_sparse_compute_so_c_ldu_count_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_so_c_ldu_fill_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_so_c_ldu_fill_f32";
    std::string_view sv(k_metal_navatala_sparse_compute_so_c_ldu_fill_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_m_i_s_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_m_i_s_f32";
    std::string_view sv(k_metal_navatala_sparse_compute_m_i_s_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_compute_m_i_s_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_compute_m_i_s_f64";
    std::string_view sv(k_metal_navatala_sparse_compute_m_i_s_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_csr_sort_after_ldu_fill_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_csr_sort_after_ldu_fill_f32";
    std::string_view sv(k_metal_navatala_sparse_csr_sort_after_ldu_fill_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

