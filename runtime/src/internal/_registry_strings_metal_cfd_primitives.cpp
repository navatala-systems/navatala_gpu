// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_cfd_primitives` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_cfd_primitives_average_face_scalar_to_cell = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_average_face_scalar_to_cell(device const float* facePhi [[buffer(0)]], device const float* magSf [[buffer(1)]], device const uint* offsets [[buffer(2)]], device const uint* faceIdx [[buffer(3)]], device const float* sumMag [[buffer(4)]], device const int* params [[buffer(5)]], device float* outCell [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    uint begU = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    float acc = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      float term = (magSf[f] * facePhi[f]);
      acc = (acc + term);
    }
    float den = sumMag[((int)(int(__gid.x)))];
    if ((den < as_type<float>(0x1e3ce508u))) {
      den = as_type<float>(0x1e3ce508u);
    }
    outCell[((int)(int(__gid.x)))] = (acc / den);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_evaluate_scalar_bc = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_evaluate_scalar_bc(device const float* internalField [[buffer(0)]], device const int* faceCells [[buffer(1)]], device const uint* bcTypeMask [[buffer(2)]], device const float* fixedValues [[buffer(3)]], device const uint* counts [[buffer(4)]], device float* boundaryOut [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(counts[0])))) {
    return;
  } else {
    uint bcType = bcTypeMask[int(__gid.x)];
    int cell = faceCells[int(__gid.x)];
    if ((((int)(bcType)) == 1)) {
      boundaryOut[int(__gid.x)] = fixedValues[int(__gid.x)];
    } else {
      boundaryOut[int(__gid.x)] = internalField[cell];
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_evaluate_vector_bc = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_evaluate_vector_bc(device const float* inX [[buffer(0)]], device const float* inY [[buffer(1)]], device const float* inZ [[buffer(2)]], device const int* faceCells [[buffer(3)]], device const uint* bcTypeMask [[buffer(4)]], device const float* fvX [[buffer(5)]], device const float* fvY [[buffer(6)]], device const float* fvZ [[buffer(7)]], device const uint* counts [[buffer(8)]], device float* outX [[buffer(9)]], device float* outY [[buffer(10)]], device float* outZ [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(counts[0])))) {
    return;
  } else {
    uint bcType = bcTypeMask[int(__gid.x)];
    int cell = faceCells[int(__gid.x)];
    if ((((int)(bcType)) == 1)) {
      outX[int(__gid.x)] = fvX[int(__gid.x)];
      outY[int(__gid.x)] = fvY[int(__gid.x)];
      outZ[int(__gid.x)] = fvZ[int(__gid.x)];
    } else {
      outX[int(__gid.x)] = inX[cell];
      outY[int(__gid.x)] = inY[cell];
      outZ[int(__gid.x)] = inZ[cell];
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_div_face_flux_csr = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_div_face_flux_csr(device const int* offsets [[buffer(0)]], device const int* faceIdx [[buffer(1)]], device const float* sign [[buffer(2)]], device const float* faceFlux [[buffer(3)]], device const float* vol [[buffer(4)]], device const int* params [[buffer(5)]], device float* outDiv [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float acc = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float term = (sign[k] * faceFlux[f]);
      acc = (acc + term);
    }
    outDiv[((int)(int(__gid.x)))] = (acc / vol[((int)(int(__gid.x)))]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_div_face_flux_vector_csr = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_div_face_flux_vector_csr(device const int* offsets [[buffer(0)]], device const int* faceIdx [[buffer(1)]], device const float* sign [[buffer(2)]], device const float* faceFluxX [[buffer(3)]], device const float* faceFluxY [[buffer(4)]], device const float* faceFluxZ [[buffer(5)]], device const float* vol [[buffer(6)]], device const int* params [[buffer(7)]], device float* outX [[buffer(8)]], device float* outY [[buffer(9)]], device float* outZ [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float accX = as_type<float>(0x00000000u);
    float accY = as_type<float>(0x00000000u);
    float accZ = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      accX = (accX + (s * faceFluxX[f]));
      accY = (accY + (s * faceFluxY[f]));
      accZ = (accZ + (s * faceFluxZ[f]));
    }
    float volCell = vol[((int)(int(__gid.x)))];
    float volSafe = (((abs(volCell) > as_type<float>(0x1e3ce508u))) ? (volCell) : (as_type<float>(0x1e3ce508u)));
    float invV = (as_type<float>(0x3f800000u) / volSafe);
    outX[((int)(int(__gid.x)))] = (accX * invV);
    outY[((int)(int(__gid.x)))] = (accY * invV);
    outZ[((int)(int(__gid.x)))] = (accZ * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2(device const float* muCell [[buffer(0)]], device const float* muBc [[buffer(1)]], device const float* gXX [[buffer(2)]], device const float* gXY [[buffer(3)]], device const float* gXZ [[buffer(4)]], device const float* gYX [[buffer(5)]], device const float* gYY [[buffer(6)]], device const float* gYZ [[buffer(7)]], device const float* gZX [[buffer(8)]], device const float* gZY [[buffer(9)]], device const float* gZZ [[buffer(10)]], device const int* owner [[buffer(11)]], device const int* neighbour [[buffer(12)]], device const float* weights [[buffer(13)]], device const float* sfX [[buffer(14)]], device const float* sfY [[buffer(15)]], device const float* sfZ [[buffer(16)]], device const int* offsets [[buffer(17)]], device const int* faceIdx [[buffer(18)]], device const float* signF [[buffer(19)]], device const float* vol [[buffer(20)]], device const int* params [[buffer(21)]], device float* outX [[buffer(22)]], device float* outY [[buffer(23)]], device float* outZ [[buffer(24)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    float sumX = as_type<float>(0x00000000u);
    float sumY = as_type<float>(0x00000000u);
    float sumZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        float s = signF[k];
        int o = owner[f];
        float muO = muCell[o];
        float goXX = gXX[o];
        float goXY = gXY[o];
        float goXZ = gXZ[o];
        float goYX = gYX[o];
        float goYY = gYY[o];
        float goYZ = gYZ[o];
        float goZX = gZX[o];
        float goZY = gZY[o];
        float goZZ = gZZ[o];
        float trO = (goXX + (goYY + goZZ));
        float twoThirdO = (as_type<float>(0x3f2aaaabu) * trO);
        float tOxx = (goXX - twoThirdO);
        float tOxy = goYX;
        float tOxz = goZX;
        float tOyx = goXY;
        float tOyy = (goYY - twoThirdO);
        float tOyz = goZY;
        float tOzx = goXZ;
        float tOzy = goYZ;
        float tOzz = (goZZ - twoThirdO);
        float tauOxx = (muO * tOxx);
        float tauOxy = (muO * tOxy);
        float tauOxz = (muO * tOxz);
        float tauOyx = (muO * tOyx);
        float tauOyy = (muO * tOyy);
        float tauOyz = (muO * tOyz);
        float tauOzx = (muO * tOzx);
        float tauOzy = (muO * tOzy);
        float tauOzz = (muO * tOzz);
        float tauFxx = tauOxx;
        float tauFxy = tauOxy;
        float tauFxz = tauOxz;
        float tauFyx = tauOyx;
        float tauFyy = tauOyy;
        float tauFyz = tauOyz;
        float tauFzx = tauOzx;
        float tauFzy = tauOzy;
        float tauFzz = tauOzz;
        int n = neighbour[f];
        if ((n >= 0)) {
          float muN = muCell[n];
          float gnXX = gXX[n];
          float gnXY = gXY[n];
          float gnXZ = gXZ[n];
          float gnYX = gYX[n];
          float gnYY = gYY[n];
          float gnYZ = gYZ[n];
          float gnZX = gZX[n];
          float gnZY = gZY[n];
          float gnZZ = gZZ[n];
          float trN = (gnXX + (gnYY + gnZZ));
          float twoThirdN = (as_type<float>(0x3f2aaaabu) * trN);
          float tNxx = (gnXX - twoThirdN);
          float tNxy = gnYX;
          float tNxz = gnZX;
          float tNyx = gnXY;
          float tNyy = (gnYY - twoThirdN);
          float tNyz = gnZY;
          float tNzx = gnXZ;
          float tNzy = gnYZ;
          float tNzz = (gnZZ - twoThirdN);
          float tauNxx = (muN * tNxx);
          float tauNxy = (muN * tNxy);
          float tauNxz = (muN * tNxz);
          float tauNyx = (muN * tNyx);
          float tauNyy = (muN * tNyy);
          float tauNyz = (muN * tNyz);
          float tauNzx = (muN * tNzx);
          float tauNzy = (muN * tNzy);
          float tauNzz = (muN * tNzz);
          float ww = weights[f];
          float iw = (as_type<float>(0x3f800000u) - ww);
          tauFxx = ((ww * tauOxx) + (iw * tauNxx));
          tauFxy = ((ww * tauOxy) + (iw * tauNxy));
          tauFxz = ((ww * tauOxz) + (iw * tauNxz));
          tauFyx = ((ww * tauOyx) + (iw * tauNyx));
          tauFyy = ((ww * tauOyy) + (iw * tauNyy));
          tauFyz = ((ww * tauOyz) + (iw * tauNyz));
          tauFzx = ((ww * tauOzx) + (iw * tauNzx));
          tauFzy = ((ww * tauOzy) + (iw * tauNzy));
          tauFzz = ((ww * tauOzz) + (iw * tauNzz));
        }
        float sx = sfX[f];
        float sy = sfY[f];
        float sz = sfZ[f];
        float fluxX = ((sx * tauFxx) + ((sy * tauFyx) + (sz * tauFzx)));
        float fluxY = ((sx * tauFxy) + ((sy * tauFyy) + (sz * tauFzy)));
        float fluxZ = ((sx * tauFxz) + ((sy * tauFyz) + (sz * tauFzz)));
        float wX = (s * fluxX);
        float wY = (s * fluxY);
        float wZ = (s * fluxZ);
        sumX = (sumX + wX);
        sumY = (sumY + wY);
        sumZ = (sumZ + wZ);
      }
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if ((v != as_type<float>(0x00000000u))) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outX[((int)(int(__gid.x)))] = (sumX * invV);
    outY[((int)(int(__gid.x)))] = (sumY * invV);
    outZ[((int)(int(__gid.x)))] = (sumZ * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_face_scalar_product = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_face_scalar_product(device const float* aAllFaces [[buffer(0)]], device const float* bAllFaces [[buffer(1)]], device const int* params [[buffer(2)]], device float* outProd [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    float a = aAllFaces[((int)(int(__gid.x)))];
    float b = bAllFaces[((int)(int(__gid.x)))];
    outProd[((int)(int(__gid.x)))] = (a * b);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_grad_vol_scalar_gauss = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_grad_vol_scalar_gauss(device const float* cellPhi [[buffer(0)]], device const float* bcValue [[buffer(1)]], device const uint* bcMask [[buffer(2)]], device const int* owner [[buffer(3)]], device const int* neighbour [[buffer(4)]], device const float* weights [[buffer(5)]], device const float* sfX [[buffer(6)]], device const float* sfY [[buffer(7)]], device const float* sfZ [[buffer(8)]], device const int* offsets [[buffer(9)]], device const int* faceIdx [[buffer(10)]], device const float* sign [[buffer(11)]], device const float* vol [[buffer(12)]], device const int* params [[buffer(13)]], device float* outX [[buffer(14)]], device float* outY [[buffer(15)]], device float* outZ [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    float sumX = as_type<float>(0x00000000u);
    float sumY = as_type<float>(0x00000000u);
    float sumZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        float s = sign[k];
        int o = owner[f];
        float po = cellPhi[o];
        float phiF = po;
        if ((f < params[2])) {
          int n = neighbour[f];
          float pn = cellPhi[n];
          float ww = weights[f];
          float iw = (as_type<float>(0x3f800000u) - ww);
          phiF = ((ww * po) + (iw * pn));
        } else {
          uint m = bcMask[f];
          if ((m == 1u)) {
            float ww = weights[f];
            float iw = (as_type<float>(0x3f800000u) - ww);
            phiF = ((ww * po) + (iw * bcValue[f]));
          }
          if ((m == 2u)) {
            phiF = bcValue[f];
          }
        }
        float w = (s * phiF);
        sumX = (sumX + (w * sfX[f]));
        sumY = (sumY + (w * sfY[f]));
        sumZ = (sumZ + (w * sfZ[f]));
      }
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if ((v != as_type<float>(0x00000000u))) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outX[((int)(int(__gid.x)))] = (sumX * invV);
    outY[((int)(int(__gid.x)))] = (sumY * invV);
    outZ[((int)(int(__gid.x)))] = (sumZ * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_grad_vol_vector_gauss = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_grad_vol_vector_gauss(device const float* cellUx [[buffer(0)]], device const float* cellUy [[buffer(1)]], device const float* cellUz [[buffer(2)]], device const float* bcX [[buffer(3)]], device const float* bcY [[buffer(4)]], device const float* bcZ [[buffer(5)]], device const uint* bcMask [[buffer(6)]], device const int* owner [[buffer(7)]], device const int* neighbour [[buffer(8)]], device const float* weights [[buffer(9)]], device const float* sfX [[buffer(10)]], device const float* sfY [[buffer(11)]], device const float* sfZ [[buffer(12)]], device const int* offsets [[buffer(13)]], device const int* faceIdx [[buffer(14)]], device const float* sign [[buffer(15)]], device const float* vol [[buffer(16)]], device const int* params [[buffer(17)]], device float* outXX [[buffer(18)]], device float* outXY [[buffer(19)]], device float* outXZ [[buffer(20)]], device float* outYX [[buffer(21)]], device float* outYY [[buffer(22)]], device float* outYZ [[buffer(23)]], device float* outZX [[buffer(24)]], device float* outZY [[buffer(25)]], device float* outZZ [[buffer(26)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    float sumXX = as_type<float>(0x00000000u);
    float sumXY = as_type<float>(0x00000000u);
    float sumXZ = as_type<float>(0x00000000u);
    float sumYX = as_type<float>(0x00000000u);
    float sumYY = as_type<float>(0x00000000u);
    float sumYZ = as_type<float>(0x00000000u);
    float sumZX = as_type<float>(0x00000000u);
    float sumZY = as_type<float>(0x00000000u);
    float sumZZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        float s = sign[k];
        int o = owner[f];
        float uxO = cellUx[o];
        float uyO = cellUy[o];
        float uzO = cellUz[o];
        float uxF = uxO;
        float uyF = uyO;
        float uzF = uzO;
        if ((f < params[2])) {
          int n = neighbour[f];
          float uxN = cellUx[n];
          float uyN = cellUy[n];
          float uzN = cellUz[n];
          float ww = weights[f];
          float iw = (as_type<float>(0x3f800000u) - ww);
          uxF = ((ww * uxO) + (iw * uxN));
          uyF = ((ww * uyO) + (iw * uyN));
          uzF = ((ww * uzO) + (iw * uzN));
        } else {
          uint m = bcMask[f];
          if ((m == 1u)) {
            float ww = weights[f];
            float iw = (as_type<float>(0x3f800000u) - ww);
            uxF = ((ww * uxO) + (iw * bcX[f]));
            uyF = ((ww * uyO) + (iw * bcY[f]));
            uzF = ((ww * uzO) + (iw * bcZ[f]));
          }
          if ((m == 2u)) {
            uxF = bcX[f];
            uyF = bcY[f];
            uzF = bcZ[f];
          }
        }
        float sfXf = sfX[f];
        float sfYf = sfY[f];
        float sfZf = sfZ[f];
        float wUx = (s * uxF);
        float wUy = (s * uyF);
        float wUz = (s * uzF);
        sumXX = (sumXX + (wUx * sfXf));
        sumXY = (sumXY + (wUx * sfYf));
        sumXZ = (sumXZ + (wUx * sfZf));
        sumYX = (sumYX + (wUy * sfXf));
        sumYY = (sumYY + (wUy * sfYf));
        sumYZ = (sumYZ + (wUy * sfZf));
        sumZX = (sumZX + (wUz * sfXf));
        sumZY = (sumZY + (wUz * sfYf));
        sumZZ = (sumZZ + (wUz * sfZf));
      }
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if ((v != as_type<float>(0x00000000u))) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outXX[((int)(int(__gid.x)))] = (sumXX * invV);
    outXY[((int)(int(__gid.x)))] = (sumXY * invV);
    outXZ[((int)(int(__gid.x)))] = (sumXZ * invV);
    outYX[((int)(int(__gid.x)))] = (sumYX * invV);
    outYY[((int)(int(__gid.x)))] = (sumYY * invV);
    outYZ[((int)(int(__gid.x)))] = (sumYZ * invV);
    outZX[((int)(int(__gid.x)))] = (sumZX * invV);
    outZY[((int)(int(__gid.x)))] = (sumZY * invV);
    outZZ[((int)(int(__gid.x)))] = (sumZZ * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_interp_scalar_face = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_interp_scalar_face(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* cellPhi [[buffer(3)]], device const float* bcValue [[buffer(4)]], device const uint* bcMask [[buffer(5)]], device const int* params [[buffer(6)]], device float* outPhiF [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float po = cellPhi[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float pn = cellPhi[n];
      float w = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w);
      outPhiF[((int)(int(__gid.x)))] = ((w * po) + (iw * pn));
    } else {
      if ((bcMask[((int)(int(__gid.x)))] == 1u)) {
        float w = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w);
        outPhiF[((int)(int(__gid.x)))] = ((w * po) + (iw * bcValue[((int)(int(__gid.x)))]));
      } else {
        if ((bcMask[((int)(int(__gid.x)))] == 2u)) {
          outPhiF[((int)(int(__gid.x)))] = bcValue[((int)(int(__gid.x)))];
        } else {
          outPhiF[((int)(int(__gid.x)))] = po;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_interp_scalar_face_all = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_interp_scalar_face_all(device const float* cellPhi [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* weights [[buffer(3)]], device const float* bcValue [[buffer(4)]], device const int* params [[buffer(5)]], device float* outPhiF [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    if ((((int)(int(__gid.x))) < params[1])) {
      int o = owner[((int)(int(__gid.x)))];
      int n = neighbour[((int)(int(__gid.x)))];
      float po = cellPhi[o];
      float pn = cellPhi[n];
      float wf = weights[((int)(int(__gid.x)))];
      if ((wf < as_type<float>(0x00000000u))) {
        wf = as_type<float>(0x00000000u);
      }
      if ((wf > as_type<float>(0x3f800000u))) {
        wf = as_type<float>(0x3f800000u);
      }
      float iw = (as_type<float>(0x3f800000u) - wf);
      float a = ((wf * po) + (iw * pn));
      if ((a < as_type<float>(0x00000000u))) {
        a = as_type<float>(0x00000000u);
      }
      if ((a > as_type<float>(0x3f800000u))) {
        a = as_type<float>(0x3f800000u);
      }
      outPhiF[((int)(int(__gid.x)))] = a;
    } else {
      float a = bcValue[((int)(int(__gid.x)))];
      if ((a < as_type<float>(0x00000000u))) {
        a = as_type<float>(0x00000000u);
      }
      if ((a > as_type<float>(0x3f800000u))) {
        a = as_type<float>(0x3f800000u);
      }
      outPhiF[((int)(int(__gid.x)))] = a;
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_interp_vector_face = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_interp_vector_face(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* cellX [[buffer(3)]], device const float* cellY [[buffer(4)]], device const float* cellZ [[buffer(5)]], device const float* bcX [[buffer(6)]], device const float* bcY [[buffer(7)]], device const float* bcZ [[buffer(8)]], device const uint* bcMask [[buffer(9)]], device const int* params [[buffer(10)]], device float* outX [[buffer(11)]], device float* outY [[buffer(12)]], device float* outZ [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float uxo = cellX[o];
    float uyo = cellY[o];
    float uzo = cellZ[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float uxn = cellX[n];
      float uyn = cellY[n];
      float uzn = cellZ[n];
      float w = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w);
      outX[((int)(int(__gid.x)))] = ((w * uxo) + (iw * uxn));
      outY[((int)(int(__gid.x)))] = ((w * uyo) + (iw * uyn));
      outZ[((int)(int(__gid.x)))] = ((w * uzo) + (iw * uzn));
    } else {
      uint m = bcMask[((int)(int(__gid.x)))];
      if ((m == 1u)) {
        float w = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w);
        outX[((int)(int(__gid.x)))] = ((w * uxo) + (iw * bcX[((int)(int(__gid.x)))]));
        outY[((int)(int(__gid.x)))] = ((w * uyo) + (iw * bcY[((int)(int(__gid.x)))]));
        outZ[((int)(int(__gid.x)))] = ((w * uzo) + (iw * bcZ[((int)(int(__gid.x)))]));
      } else {
        if ((m == 2u)) {
          outX[((int)(int(__gid.x)))] = bcX[((int)(int(__gid.x)))];
          outY[((int)(int(__gid.x)))] = bcY[((int)(int(__gid.x)))];
          outZ[((int)(int(__gid.x)))] = bcZ[((int)(int(__gid.x)))];
        } else {
          outX[((int)(int(__gid.x)))] = uxo;
          outY[((int)(int(__gid.x)))] = uyo;
          outZ[((int)(int(__gid.x)))] = uzo;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_linear_upwind_scalar_face = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_linear_upwind_scalar_face(device const float* flux [[buffer(0)]], device const float* xCell [[buffer(1)]], device const float* gradX [[buffer(2)]], device const float* gradY [[buffer(3)]], device const float* gradZ [[buffer(4)]], device const float* limiter [[buffer(5)]], device const int* owner [[buffer(6)]], device const int* nei [[buffer(7)]], device const float* cx [[buffer(8)]], device const float* cy [[buffer(9)]], device const float* cz [[buffer(10)]], device const float* cfx [[buffer(11)]], device const float* cfy [[buffer(12)]], device const float* cfz [[buffer(13)]], device const float* bcValue [[buffer(14)]], device const uint* bcMask [[buffer(15)]], device const uint* params [[buffer(16)]], device float* outFace [[buffer(17)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(params[0])))) {
    return;
  } else {
    float phif = flux[int(__gid.x)];
    int own = owner[int(__gid.x)];
    if ((int(__gid.x) < ((int)(params[1])))) {
      if ((phif >= as_type<float>(0x00000000u))) {
        float dx = (cfx[int(__gid.x)] - cx[own]);
        float dy = (cfy[int(__gid.x)] - cy[own]);
        float dz = (cfz[int(__gid.x)] - cz[own]);
        float corr = (((dx * gradX[own]) + (dy * gradY[own])) + (dz * gradZ[own]));
        outFace[int(__gid.x)] = (xCell[own] + (limiter[own] * corr));
      } else {
        int neiCell = nei[int(__gid.x)];
        float dx = (cfx[int(__gid.x)] - cx[neiCell]);
        float dy = (cfy[int(__gid.x)] - cy[neiCell]);
        float dz = (cfz[int(__gid.x)] - cz[neiCell]);
        float corr = (((dx * gradX[neiCell]) + (dy * gradY[neiCell])) + (dz * gradZ[neiCell]));
        outFace[int(__gid.x)] = (xCell[neiCell] + (limiter[neiCell] * corr));
      }
    } else {
      if ((phif >= as_type<float>(0x00000000u))) {
        outFace[int(__gid.x)] = xCell[own];
      } else {
        uint mask = bcMask[int(__gid.x)];
        if ((((int)(mask)) == 0)) {
          outFace[int(__gid.x)] = xCell[own];
        } else {
          outFace[int(__gid.x)] = bcValue[int(__gid.x)];
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_phi_from_u = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_phi_from_u(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* ux [[buffer(6)]], device const float* uy [[buffer(7)]], device const float* uz [[buffer(8)]], device const float* bcX [[buffer(9)]], device const float* bcY [[buffer(10)]], device const float* bcZ [[buffer(11)]], device const uint* bcMask [[buffer(12)]], device const int* params [[buffer(13)]], device float* outPhi [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float w = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w);
      float ufx = ((w * uox) + (iw * ux[n]));
      float ufy = ((w * uoy) + (iw * uy[n]));
      float ufz = ((w * uoz) + (iw * uz[n]));
      outPhi[((int)(int(__gid.x)))] = (((sfX[((int)(int(__gid.x)))] * ufx) + (sfY[((int)(int(__gid.x)))] * ufy)) + (sfZ[((int)(int(__gid.x)))] * ufz));
    } else {
      if ((bcMask[((int)(int(__gid.x)))] == 1u)) {
        float w = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w);
        float ufx = ((w * uox) + (iw * bcX[((int)(int(__gid.x)))]));
        float ufy = ((w * uoy) + (iw * bcY[((int)(int(__gid.x)))]));
        float ufz = ((w * uoz) + (iw * bcZ[((int)(int(__gid.x)))]));
        outPhi[((int)(int(__gid.x)))] = (((sfX[((int)(int(__gid.x)))] * ufx) + (sfY[((int)(int(__gid.x)))] * ufy)) + (sfZ[((int)(int(__gid.x)))] * ufz));
      } else {
        if ((bcMask[((int)(int(__gid.x)))] == 2u)) {
          float ufx = bcX[((int)(int(__gid.x)))];
          float ufy = bcY[((int)(int(__gid.x)))];
          float ufz = bcZ[((int)(int(__gid.x)))];
          outPhi[((int)(int(__gid.x)))] = (((sfX[((int)(int(__gid.x)))] * ufx) + (sfY[((int)(int(__gid.x)))] * ufy)) + (sfZ[((int)(int(__gid.x)))] * ufz));
        } else {
          outPhi[((int)(int(__gid.x)))] = (((sfX[((int)(int(__gid.x)))] * uox) + (sfY[((int)(int(__gid.x)))] * uoy)) + (sfZ[((int)(int(__gid.x)))] * uoz));
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_phi_from_u_boundary = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_phi_from_u_boundary(device const int* owner [[buffer(0)]], device const float* weights [[buffer(1)]], device const float* sfX [[buffer(2)]], device const float* sfY [[buffer(3)]], device const float* sfZ [[buffer(4)]], device const float* ux [[buffer(5)]], device const float* uy [[buffer(6)]], device const float* uz [[buffer(7)]], device const float* bcx [[buffer(8)]], device const float* bcy [[buffer(9)]], device const float* bcz [[buffer(10)]], device const uint* bcmask [[buffer(11)]], device const uint* params [[buffer(12)]], device float* outPhi [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int f = (((int)(params[1])) + int(__gid.x));
  if ((f >= ((int)(params[0])))) {
    return;
  } else {
    int o = owner[f];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    float ufx = uox;
    float ufy = uoy;
    float ufz = uoz;
    uint m = bcmask[f];
    int mInt = ((int)(m));
    if ((mInt == 1)) {
      float w = weights[f];
      float iw = (as_type<float>(0x3f800000u) - w);
      ufx = ((w * uox) + (iw * bcx[f]));
      ufy = ((w * uoy) + (iw * bcy[f]));
      ufz = ((w * uoz) + (iw * bcz[f]));
    } else {
      if ((mInt == 2)) {
        ufx = bcx[f];
        ufy = bcy[f];
        ufz = bcz[f];
      }
    }
    outPhi[f] = (((sfX[f] * ufx) + (sfY[f] * ufy)) + (sfZ[f] * ufz));
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_phi_from_u_internal = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_phi_from_u_internal(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* ux [[buffer(6)]], device const float* uy [[buffer(7)]], device const float* uz [[buffer(8)]], device const uint* params [[buffer(9)]], device float* outPhi [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(params[1])))) {
    return;
  } else {
    int o = owner[int(__gid.x)];
    int n = neighbour[int(__gid.x)];
    float w = weights[int(__gid.x)];
    float iw = (as_type<float>(0x3f800000u) - w);
    float ufx = ((w * ux[o]) + (iw * ux[n]));
    float ufy = ((w * uy[o]) + (iw * uy[n]));
    float ufz = ((w * uz[o]) + (iw * uz[n]));
    outPhi[int(__gid.x)] = (((sfX[int(__gid.x)] * ufx) + (sfY[int(__gid.x)] * ufy)) + (sfZ[int(__gid.x)] * ufz));
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_phig_gravity = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_phig_gravity(device const float* ghf [[buffer(0)]], device const float* snGradRho [[buffer(1)]], device const float* rAUf [[buffer(2)]], device const float* magSf [[buffer(3)]], device const uint* params [[buffer(4)]], device float* outPhig [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(params[0])))) {
    return;
  } else {
    outPhig[int(__gid.x)] = (-((ghf[int(__gid.x)] * snGradRho[int(__gid.x)]) * (rAUf[int(__gid.x)] * magSf[int(__gid.x)])));
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_r_a_u_from_dt_rho(device const float* rho [[buffer(0)]], device const float* paramsF [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outRAU [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)(int(__gid.x)))];
    outRAU[((int)(int(__gid.x)))] = (paramsF[0] / r);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_integrated(device const float* rho [[buffer(0)]], device const float* vol [[buffer(1)]], device const float* paramsF [[buffer(2)]], device const int* counts [[buffer(3)]], device float* outRAU [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)(int(__gid.x)))];
    float v = vol[((int)(int(__gid.x)))];
    float denom = (r * v);
    if ((denom != as_type<float>(0x00000000u))) {
      outRAU[((int)(int(__gid.x)))] = (paramsF[0] / denom);
    } else {
      outRAU[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_sp(device const float* rho [[buffer(0)]], device const float* sp [[buffer(1)]], device const float* paramsF [[buffer(2)]], device const int* counts [[buffer(3)]], device float* outRAU [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)(int(__gid.x)))];
    float s = sp[((int)(int(__gid.x)))];
    float denom = (r + (paramsF[0] * s));
    outRAU[((int)(int(__gid.x)))] = (paramsF[0] / denom);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated(device const float* rho [[buffer(0)]], device const float* sp [[buffer(1)]], device const float* vol [[buffer(2)]], device const float* paramsF [[buffer(3)]], device const int* counts [[buffer(4)]], device float* outRAU [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)(int(__gid.x)))];
    float s = sp[((int)(int(__gid.x)))];
    float v = vol[((int)(int(__gid.x)))];
    float denom = ((r + (paramsF[0] * s)) * v);
    if ((denom != as_type<float>(0x00000000u))) {
      outRAU[((int)(int(__gid.x)))] = (paramsF[0] / denom);
    } else {
      outRAU[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_r_a_u_from_u_eqn_approx(device const float* rhoCell [[buffer(0)]], device const float* muCell [[buffer(1)]], device const float* phiFaces [[buffer(2)]], device const float* rhoBcVal [[buffer(3)]], device const uint* rhoBcMask [[buffer(4)]], device const float* muBcVal [[buffer(5)]], device const uint* muBcMask [[buffer(6)]], device const int* ownerAll [[buffer(7)]], device const int* neighbourInt [[buffer(8)]], device const float* weightsAll [[buffer(9)]], device const float* magSfAll [[buffer(10)]], device const float* deltaAll [[buffer(11)]], device const uint* offsets [[buffer(12)]], device const uint* faceIdx [[buffer(13)]], device const int* sign [[buffer(14)]], device const float* volCells [[buffer(15)]], device const int* counts3 [[buffer(16)]], device const float* paramsF [[buffer(17)]], device float* outRAU [[buffer(18)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts3[0])) {
    return;
  } else {
    float invDt = (as_type<float>(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)(int(__gid.x)))];
    float A = (rhoCell[((int)(int(__gid.x)))] * invDt);
    float conv = as_type<float>(0x00000000u);
    float diff = as_type<float>(0x00000000u);
    uint begU = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      if ((f >= counts3[1])) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = as_type<float>(0x00000000u);
        float muF = as_type<float>(0x00000000u);
        if ((f < counts3[2])) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (as_type<float>(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if ((rhoBcMask[f] != 0u)) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if ((muBcMask[f] != 0u)) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if ((s < 0)) {
          outFlux = (as_type<float>(0x00000000u) - rhoPhi);
        }
        if ((outFlux > as_type<float>(0x00000000u))) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    if ((A != as_type<float>(0x00000000u))) {
      outRAU[((int)(int(__gid.x)))] = (as_type<float>(0x3f800000u) / A);
    } else {
      outRAU[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp(device const float* rhoCell [[buffer(0)]], device const float* muCell [[buffer(1)]], device const float* phiFaces [[buffer(2)]], device const float* rhoBcVal [[buffer(3)]], device const uint* rhoBcMask [[buffer(4)]], device const float* muBcVal [[buffer(5)]], device const uint* muBcMask [[buffer(6)]], device const int* ownerAll [[buffer(7)]], device const int* neighbourInt [[buffer(8)]], device const float* weightsAll [[buffer(9)]], device const float* magSfAll [[buffer(10)]], device const float* deltaAll [[buffer(11)]], device const uint* offsets [[buffer(12)]], device const uint* faceIdx [[buffer(13)]], device const int* sign [[buffer(14)]], device const float* volCells [[buffer(15)]], device const float* sp [[buffer(16)]], device const int* counts3 [[buffer(17)]], device const float* paramsF [[buffer(18)]], device float* outRAU [[buffer(19)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts3[0])) {
    return;
  } else {
    float invDt = (as_type<float>(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)(int(__gid.x)))];
    float A = (rhoCell[((int)(int(__gid.x)))] * invDt);
    float conv = as_type<float>(0x00000000u);
    float diff = as_type<float>(0x00000000u);
    uint begU = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      if ((f >= counts3[1])) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = as_type<float>(0x00000000u);
        float muF = as_type<float>(0x00000000u);
        if ((f < counts3[2])) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (as_type<float>(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if ((rhoBcMask[f] != 0u)) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if ((muBcMask[f] != 0u)) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if ((s < 0)) {
          outFlux = (as_type<float>(0x00000000u) - rhoPhi);
        }
        if ((outFlux > as_type<float>(0x00000000u))) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    A = (A + sp[((int)(int(__gid.x)))]);
    if ((A != as_type<float>(0x00000000u))) {
      outRAU[((int)(int(__gid.x)))] = (as_type<float>(0x3f800000u) / A);
    } else {
      outRAU[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_rho_from_alpha = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_rho_from_alpha(device const float* alpha [[buffer(0)]], device const float* paramsF [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outRho [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float a = alpha[((int)(int(__gid.x)))];
    if ((a < as_type<float>(0x00000000u))) {
      a = as_type<float>(0x00000000u);
    }
    if ((a > as_type<float>(0x3f800000u))) {
      a = as_type<float>(0x3f800000u);
    }
    float invA = (as_type<float>(0x3f800000u) - a);
    float r = ((a * paramsF[0]) + (invA * paramsF[1]));
    outRho[((int)(int(__gid.x)))] = r;
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_rho_phi_from_rho_and_phi = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_rho_phi_from_rho_and_phi(device const float* phiAll [[buffer(0)]], device const float* rhoCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* weights [[buffer(4)]], device const float* rhoBcVal [[buffer(5)]], device const uint* rhoBcMask [[buffer(6)]], device const int* params [[buffer(7)]], device float* outRhoPhi [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float rhoF = rhoCell[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float w0 = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w0);
      rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
    } else {
      uint m = rhoBcMask[((int)(int(__gid.x)))];
      if ((m == 1u)) {
        float w0 = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w0);
        rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[((int)(int(__gid.x)))]));
      }
      if ((m == 2u)) {
        rhoF = rhoBcVal[((int)(int(__gid.x)))];
      }
    }
    float flux = phiAll[((int)(int(__gid.x)))];
    outRhoPhi[((int)(int(__gid.x)))] = (rhoF * flux);
  }
}

)kernel";
const char* k_metal_navatala_cfd_primitives_sn_grad_scalar = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_sn_grad_scalar(device const float* cellVal [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* delta [[buffer(3)]], device const float* faceVal [[buffer(4)]], device const uint* faceMask [[buffer(5)]], device const int* params [[buffer(6)]], device float* outSnGrad [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float aO = cellVal[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float aN = cellVal[n];
      outSnGrad[((int)(int(__gid.x)))] = (delta[((int)(int(__gid.x)))] * (aN - aO));
    } else {
      if ((faceMask[((int)(int(__gid.x)))] != 0u)) {
        float aN = faceVal[((int)(int(__gid.x)))];
        outSnGrad[((int)(int(__gid.x)))] = (delta[((int)(int(__gid.x)))] * (aN - aO));
      } else {
        outSnGrad[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
      }
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_average_face_scalar_to_cell[] = {
  { "facePhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumMag", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCell", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_average_face_scalar_to_cell = {
  1,
  "navatala_cfd_primitives_average_face_scalar_to_cell",
  "metal",
  "navatala_cfd_primitives_average_face_scalar_to_cell",
  "kernel:metal:navatala_cfd_primitives_average_face_scalar_to_cell",
  "abi-r1:metal:navatala_cfd_primitives_average_face_scalar_to_cell",
  "abi-r1:metal:navatala_cfd_primitives_average_face_scalar_to_cell",
  7,
  kAbiArgs_metal_navatala_cfd_primitives_average_face_scalar_to_cell
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_evaluate_scalar_bc[] = {
  { "internalField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCells", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcTypeMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedValues", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "boundaryOut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_evaluate_scalar_bc = {
  1,
  "navatala_cfd_primitives_evaluate_scalar_bc",
  "metal",
  "navatala_cfd_primitives_evaluate_scalar_bc",
  "kernel:metal:navatala_cfd_primitives_evaluate_scalar_bc",
  "abi-r1:metal:navatala_cfd_primitives_evaluate_scalar_bc",
  "abi-r1:metal:navatala_cfd_primitives_evaluate_scalar_bc",
  6,
  kAbiArgs_metal_navatala_cfd_primitives_evaluate_scalar_bc
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_evaluate_vector_bc[] = {
  { "inX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcTypeMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fvX", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fvY", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fvZ", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_evaluate_vector_bc = {
  1,
  "navatala_cfd_primitives_evaluate_vector_bc",
  "metal",
  "navatala_cfd_primitives_evaluate_vector_bc",
  "kernel:metal:navatala_cfd_primitives_evaluate_vector_bc",
  "abi-r1:metal:navatala_cfd_primitives_evaluate_vector_bc",
  "abi-r1:metal:navatala_cfd_primitives_evaluate_vector_bc",
  12,
  kAbiArgs_metal_navatala_cfd_primitives_evaluate_vector_bc
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_div_face_flux_csr[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFlux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiv", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_div_face_flux_csr = {
  1,
  "navatala_cfd_primitives_div_face_flux_csr",
  "metal",
  "navatala_cfd_primitives_div_face_flux_csr",
  "kernel:metal:navatala_cfd_primitives_div_face_flux_csr",
  "abi-r1:metal:navatala_cfd_primitives_div_face_flux_csr",
  "abi-r1:metal:navatala_cfd_primitives_div_face_flux_csr",
  7,
  kAbiArgs_metal_navatala_cfd_primitives_div_face_flux_csr
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_div_face_flux_vector_csr[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_div_face_flux_vector_csr = {
  1,
  "navatala_cfd_primitives_div_face_flux_vector_csr",
  "metal",
  "navatala_cfd_primitives_div_face_flux_vector_csr",
  "kernel:metal:navatala_cfd_primitives_div_face_flux_vector_csr",
  "abi-r1:metal:navatala_cfd_primitives_div_face_flux_vector_csr",
  "abi-r1:metal:navatala_cfd_primitives_div_face_flux_vector_csr",
  11,
  kAbiArgs_metal_navatala_cfd_primitives_div_face_flux_vector_csr
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBc", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXX", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXY", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXZ", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYX", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYY", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYZ", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZX", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZY", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZZ", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signF", 19, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 20, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 21, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outX", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2 = {
  1,
  "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "metal",
  "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "kernel:metal:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "abi-r1:metal:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "abi-r1:metal:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  25,
  kAbiArgs_metal_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_face_scalar_product[] = {
  { "aAllFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bAllFaces", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outProd", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_face_scalar_product = {
  1,
  "navatala_cfd_primitives_face_scalar_product",
  "metal",
  "navatala_cfd_primitives_face_scalar_product",
  "kernel:metal:navatala_cfd_primitives_face_scalar_product",
  "abi-r1:metal:navatala_cfd_primitives_face_scalar_product",
  "abi-r1:metal:navatala_cfd_primitives_face_scalar_product",
  4,
  kAbiArgs_metal_navatala_cfd_primitives_face_scalar_product
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_grad_vol_scalar_gauss[] = {
  { "cellPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outX", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_grad_vol_scalar_gauss = {
  1,
  "navatala_cfd_primitives_grad_vol_scalar_gauss",
  "metal",
  "navatala_cfd_primitives_grad_vol_scalar_gauss",
  "kernel:metal:navatala_cfd_primitives_grad_vol_scalar_gauss",
  "abi-r1:metal:navatala_cfd_primitives_grad_vol_scalar_gauss",
  "abi-r1:metal:navatala_cfd_primitives_grad_vol_scalar_gauss",
  17,
  kAbiArgs_metal_navatala_cfd_primitives_grad_vol_scalar_gauss
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_grad_vol_vector_gauss[] = {
  { "cellUx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellUy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellUz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outXX", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outXY", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outXZ", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outYX", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outYY", 22, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outYZ", 23, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZX", 24, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZY", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZZ", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_grad_vol_vector_gauss = {
  1,
  "navatala_cfd_primitives_grad_vol_vector_gauss",
  "metal",
  "navatala_cfd_primitives_grad_vol_vector_gauss",
  "kernel:metal:navatala_cfd_primitives_grad_vol_vector_gauss",
  "abi-r1:metal:navatala_cfd_primitives_grad_vol_vector_gauss",
  "abi-r1:metal:navatala_cfd_primitives_grad_vol_vector_gauss",
  27,
  kAbiArgs_metal_navatala_cfd_primitives_grad_vol_vector_gauss
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_interp_scalar_face[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellPhi", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhiF", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_interp_scalar_face = {
  1,
  "navatala_cfd_primitives_interp_scalar_face",
  "metal",
  "navatala_cfd_primitives_interp_scalar_face",
  "kernel:metal:navatala_cfd_primitives_interp_scalar_face",
  "abi-r1:metal:navatala_cfd_primitives_interp_scalar_face",
  "abi-r1:metal:navatala_cfd_primitives_interp_scalar_face",
  8,
  kAbiArgs_metal_navatala_cfd_primitives_interp_scalar_face
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_interp_scalar_face_all[] = {
  { "cellPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhiF", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_interp_scalar_face_all = {
  1,
  "navatala_cfd_primitives_interp_scalar_face_all",
  "metal",
  "navatala_cfd_primitives_interp_scalar_face_all",
  "kernel:metal:navatala_cfd_primitives_interp_scalar_face_all",
  "abi-r1:metal:navatala_cfd_primitives_interp_scalar_face_all",
  "abi-r1:metal:navatala_cfd_primitives_interp_scalar_face_all",
  7,
  kAbiArgs_metal_navatala_cfd_primitives_interp_scalar_face_all
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_interp_vector_face[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outX", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_interp_vector_face = {
  1,
  "navatala_cfd_primitives_interp_vector_face",
  "metal",
  "navatala_cfd_primitives_interp_vector_face",
  "kernel:metal:navatala_cfd_primitives_interp_vector_face",
  "abi-r1:metal:navatala_cfd_primitives_interp_vector_face",
  "abi-r1:metal:navatala_cfd_primitives_interp_vector_face",
  14,
  kAbiArgs_metal_navatala_cfd_primitives_interp_vector_face
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_linear_upwind_scalar_face[] = {
  { "flux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "xCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradX", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradY", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradZ", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "limiter", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cx", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cy", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cz", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfx", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfy", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cfz", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outFace", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_linear_upwind_scalar_face = {
  1,
  "navatala_cfd_primitives_linear_upwind_scalar_face",
  "metal",
  "navatala_cfd_primitives_linear_upwind_scalar_face",
  "kernel:metal:navatala_cfd_primitives_linear_upwind_scalar_face",
  "abi-r1:metal:navatala_cfd_primitives_linear_upwind_scalar_face",
  "abi-r1:metal:navatala_cfd_primitives_linear_upwind_scalar_face",
  18,
  kAbiArgs_metal_navatala_cfd_primitives_linear_upwind_scalar_face
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_phi_from_u[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhi", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_phi_from_u = {
  1,
  "navatala_cfd_primitives_phi_from_u",
  "metal",
  "navatala_cfd_primitives_phi_from_u",
  "kernel:metal:navatala_cfd_primitives_phi_from_u",
  "abi-r1:metal:navatala_cfd_primitives_phi_from_u",
  "abi-r1:metal:navatala_cfd_primitives_phi_from_u",
  15,
  kAbiArgs_metal_navatala_cfd_primitives_phi_from_u
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_phi_from_u_boundary[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcx", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcy", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcz", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcmask", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhi", 13, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_phi_from_u_boundary = {
  1,
  "navatala_cfd_primitives_phi_from_u_boundary",
  "metal",
  "navatala_cfd_primitives_phi_from_u_boundary",
  "kernel:metal:navatala_cfd_primitives_phi_from_u_boundary",
  "abi-r1:metal:navatala_cfd_primitives_phi_from_u_boundary",
  "abi-r1:metal:navatala_cfd_primitives_phi_from_u_boundary",
  14,
  kAbiArgs_metal_navatala_cfd_primitives_phi_from_u_boundary
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_phi_from_u_internal[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhi", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_phi_from_u_internal = {
  1,
  "navatala_cfd_primitives_phi_from_u_internal",
  "metal",
  "navatala_cfd_primitives_phi_from_u_internal",
  "kernel:metal:navatala_cfd_primitives_phi_from_u_internal",
  "abi-r1:metal:navatala_cfd_primitives_phi_from_u_internal",
  "abi-r1:metal:navatala_cfd_primitives_phi_from_u_internal",
  11,
  kAbiArgs_metal_navatala_cfd_primitives_phi_from_u_internal
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_phig_gravity[] = {
  { "ghf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "snGradRho", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhig", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_phig_gravity = {
  1,
  "navatala_cfd_primitives_phig_gravity",
  "metal",
  "navatala_cfd_primitives_phig_gravity",
  "kernel:metal:navatala_cfd_primitives_phig_gravity",
  "abi-r1:metal:navatala_cfd_primitives_phig_gravity",
  "abi-r1:metal:navatala_cfd_primitives_phig_gravity",
  6,
  kAbiArgs_metal_navatala_cfd_primitives_phig_gravity
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho",
  "metal",
  "navatala_cfd_primitives_r_a_u_from_dt_rho",
  "kernel:metal:navatala_cfd_primitives_r_a_u_from_dt_rho",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho",
  4,
  kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "metal",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "kernel:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  5,
  kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "metal",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "kernel:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  5,
  kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "metal",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "kernel:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  6,
  kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx[] = {
  { "rhoCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBcVal", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBcMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerAll", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbourInt", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weightsAll", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSfAll", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaAll", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "volCells", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts3", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx = {
  1,
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "metal",
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "kernel:metal:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  19,
  kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp[] = {
  { "rhoCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBcVal", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBcMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerAll", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbourInt", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weightsAll", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSfAll", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaAll", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "volCells", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts3", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp = {
  1,
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "metal",
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "kernel:metal:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "abi-r1:metal:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  20,
  kAbiArgs_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_rho_from_alpha[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRho", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_rho_from_alpha = {
  1,
  "navatala_cfd_primitives_rho_from_alpha",
  "metal",
  "navatala_cfd_primitives_rho_from_alpha",
  "kernel:metal:navatala_cfd_primitives_rho_from_alpha",
  "abi-r1:metal:navatala_cfd_primitives_rho_from_alpha",
  "abi-r1:metal:navatala_cfd_primitives_rho_from_alpha",
  4,
  kAbiArgs_metal_navatala_cfd_primitives_rho_from_alpha
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_rho_phi_from_rho_and_phi[] = {
  { "phiAll", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outRhoPhi", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_rho_phi_from_rho_and_phi = {
  1,
  "navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "metal",
  "navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "kernel:metal:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "abi-r1:metal:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "abi-r1:metal:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  9,
  kAbiArgs_metal_navatala_cfd_primitives_rho_phi_from_rho_and_phi
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_primitives_sn_grad_scalar[] = {
  { "cellVal", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceVal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSnGrad", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_primitives_sn_grad_scalar = {
  1,
  "navatala_cfd_primitives_sn_grad_scalar",
  "metal",
  "navatala_cfd_primitives_sn_grad_scalar",
  "kernel:metal:navatala_cfd_primitives_sn_grad_scalar",
  "abi-r1:metal:navatala_cfd_primitives_sn_grad_scalar",
  "abi-r1:metal:navatala_cfd_primitives_sn_grad_scalar",
  8,
  kAbiArgs_metal_navatala_cfd_primitives_sn_grad_scalar
};

bool tryGetKernelAbiManifest_metal_cfd_primitives(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_average_face_scalar_to_cell") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_average_face_scalar_to_cell;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_evaluate_scalar_bc") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_evaluate_scalar_bc;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_evaluate_vector_bc") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_evaluate_vector_bc;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_div_face_flux_csr") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_div_face_flux_csr;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_div_face_flux_vector_csr") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_div_face_flux_vector_csr;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_face_scalar_product") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_face_scalar_product;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_grad_vol_scalar_gauss") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_grad_vol_scalar_gauss;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_grad_vol_vector_gauss") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_grad_vol_vector_gauss;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_interp_scalar_face") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_interp_scalar_face;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_interp_scalar_face_all") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_interp_scalar_face_all;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_interp_vector_face") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_interp_vector_face;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_linear_upwind_scalar_face") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_linear_upwind_scalar_face;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phi_from_u") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_phi_from_u;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phi_from_u_boundary") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_phi_from_u_boundary;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phi_from_u_internal") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_phi_from_u_internal;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phig_gravity") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_phig_gravity;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_rho_from_alpha") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_rho_from_alpha;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_rho_phi_from_rho_and_phi") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_rho_phi_from_rho_and_phi;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_sn_grad_scalar") {
    out = &kAbiManifest_metal_navatala_cfd_primitives_sn_grad_scalar;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_cfd_primitives(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_average_face_scalar_to_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_average_face_scalar_to_cell";
    std::string_view sv(k_metal_navatala_cfd_primitives_average_face_scalar_to_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_evaluate_scalar_bc") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_evaluate_scalar_bc";
    std::string_view sv(k_metal_navatala_cfd_primitives_evaluate_scalar_bc);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_evaluate_vector_bc") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_evaluate_vector_bc";
    std::string_view sv(k_metal_navatala_cfd_primitives_evaluate_vector_bc);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_div_face_flux_csr") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_csr";
    std::string_view sv(k_metal_navatala_cfd_primitives_div_face_flux_csr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_div_face_flux_vector_csr") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_vector_csr";
    std::string_view sv(k_metal_navatala_cfd_primitives_div_face_flux_vector_csr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2";
    std::string_view sv(k_metal_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_face_scalar_product") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_face_scalar_product";
    std::string_view sv(k_metal_navatala_cfd_primitives_face_scalar_product);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_grad_vol_scalar_gauss") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_grad_vol_scalar_gauss";
    std::string_view sv(k_metal_navatala_cfd_primitives_grad_vol_scalar_gauss);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_grad_vol_vector_gauss") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_grad_vol_vector_gauss";
    std::string_view sv(k_metal_navatala_cfd_primitives_grad_vol_vector_gauss);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_interp_scalar_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_interp_scalar_face";
    std::string_view sv(k_metal_navatala_cfd_primitives_interp_scalar_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_interp_scalar_face_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_interp_scalar_face_all";
    std::string_view sv(k_metal_navatala_cfd_primitives_interp_scalar_face_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_interp_vector_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_interp_vector_face";
    std::string_view sv(k_metal_navatala_cfd_primitives_interp_vector_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_linear_upwind_scalar_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_linear_upwind_scalar_face";
    std::string_view sv(k_metal_navatala_cfd_primitives_linear_upwind_scalar_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phi_from_u") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u";
    std::string_view sv(k_metal_navatala_cfd_primitives_phi_from_u);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phi_from_u_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u_boundary";
    std::string_view sv(k_metal_navatala_cfd_primitives_phi_from_u_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phi_from_u_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u_internal";
    std::string_view sv(k_metal_navatala_cfd_primitives_phi_from_u_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_phig_gravity") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_phig_gravity";
    std::string_view sv(k_metal_navatala_cfd_primitives_phig_gravity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho";
    std::string_view sv(k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated";
    std::string_view sv(k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_sp";
    std::string_view sv(k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated";
    std::string_view sv(k_metal_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_u_eqn_approx";
    std::string_view sv(k_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp";
    std::string_view sv(k_metal_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_rho_from_alpha") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_rho_from_alpha";
    std::string_view sv(k_metal_navatala_cfd_primitives_rho_from_alpha);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_rho_phi_from_rho_and_phi") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_rho_phi_from_rho_and_phi";
    std::string_view sv(k_metal_navatala_cfd_primitives_rho_phi_from_rho_and_phi);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_primitives_sn_grad_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_primitives_sn_grad_scalar";
    std::string_view sv(k_metal_navatala_cfd_primitives_sn_grad_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

