// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_cfd_primitives` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_navatala_cfd_primitives_average_face_scalar_to_cell = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_average_face_scalar_to_cell(const float* facePhi, const float* magSf, const unsigned int* offsets, const unsigned int* faceIdx, const float* sumMag, const int* params, float* outCell) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    unsigned int begU = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    unsigned int endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    float acc = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int fU = faceIdx[k];
      int f = ((int)(fU));
      float term = (magSf[f] * facePhi[f]);
      acc = (acc + term);
    }
    float den = sumMag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (den < __uint_as_float(0x1e3ce508u)) {
      den = __uint_as_float(0x1e3ce508u);
    }
    outCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (acc / den);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_evaluate_scalar_bc = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_evaluate_scalar_bc(const float* internalField, const int* faceCells, const unsigned int* bcTypeMask, const float* fixedValues, const unsigned int* counts, float* boundaryOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    unsigned int bcType = bcTypeMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int cell = faceCells[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (((int)(bcType)) == 1) {
      boundaryOut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = fixedValues[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    } else {
      boundaryOut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = internalField[cell];
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_evaluate_vector_bc = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_evaluate_vector_bc(const float* inX, const float* inY, const float* inZ, const int* faceCells, const unsigned int* bcTypeMask, const float* fvX, const float* fvY, const float* fvZ, const unsigned int* counts, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    unsigned int bcType = bcTypeMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int cell = faceCells[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (((int)(bcType)) == 1) {
      outX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = fvX[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      outY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = fvY[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      outZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = fvZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    } else {
      outX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = inX[cell];
      outY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = inY[cell];
      outZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = inZ[cell];
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_div_face_flux_csr = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_div_face_flux_csr(const int* offsets, const int* faceIdx, const float* sign, const float* faceFlux, const float* vol, const int* params, float* outDiv) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float acc = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float term = (sign[k] * faceFlux[f]);
      acc = (acc + term);
    }
    outDiv[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (acc / vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_div_face_flux_csr_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_div_face_flux_csr_f64(const int* offsets, const int* faceIdx, const float* sign, const double* faceFlux, const double* vol, const int* params, double* outDiv) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double term = (signV * faceFlux[f]);
      acc = (acc + term);
    }
    outDiv[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (acc / ((double)(vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_div_face_flux_vector_csr = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_div_face_flux_vector_csr(const int* offsets, const int* faceIdx, const float* sign, const float* faceFluxX, const float* faceFluxY, const float* faceFluxZ, const float* vol, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float accX = __uint_as_float(0x00000000u);
    float accY = __uint_as_float(0x00000000u);
    float accZ = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      accX = (accX + (s * faceFluxX[f]));
      accY = (accY + (s * faceFluxY[f]));
      accZ = (accZ + (s * faceFluxZ[f]));
    }
    float volCell = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float volSafe = (((abs(volCell) > __uint_as_float(0x1e3ce508u))) ? (volCell) : (__uint_as_float(0x1e3ce508u)));
    float invV = (__uint_as_float(0x3f800000u) / volSafe);
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (accX * invV);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (accY * invV);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (accZ * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2(const float* muCell, const float* muBc, const float* gXX, const float* gXY, const float* gXZ, const float* gYX, const float* gYY, const float* gYZ, const float* gZX, const float* gZY, const float* gZZ, const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* signF, const float* vol, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float sumX = __uint_as_float(0x00000000u);
    float sumY = __uint_as_float(0x00000000u);
    float sumZ = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
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
        float twoThirdO = (__uint_as_float(0x3f2aaaabu) * trO);
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
        if (n >= 0) {
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
          float twoThirdN = (__uint_as_float(0x3f2aaaabu) * trN);
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
          float iw = (__uint_as_float(0x3f800000u) - ww);
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
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumX * invV);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumY * invV);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZ * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_face_scalar_product = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_face_scalar_product(const float* aAllFaces, const float* bAllFaces, const int* params, float* outProd) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float a = aAllFaces[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float b = bAllFaces[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outProd[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (a * b);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_grad_vol_scalar_gauss = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_grad_vol_scalar_gauss(const float* cellPhi, const float* bcValue, const unsigned int* bcMask, const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float sumX = __uint_as_float(0x00000000u);
    float sumY = __uint_as_float(0x00000000u);
    float sumZ = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        float s = sign[k];
        int o = owner[f];
        float po = cellPhi[o];
        float phiF = po;
        if (f < params[2]) {
          int n = neighbour[f];
          float pn = cellPhi[n];
          float ww = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - ww);
          phiF = ((ww * po) + (iw * pn));
        } else {
          unsigned int m = bcMask[f];
          if (m == 1u) {
            float ww = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - ww);
            phiF = ((ww * po) + (iw * bcValue[f]));
          }
          if (m == 2u) {
            phiF = bcValue[f];
          }
        }
        float w = (s * phiF);
        sumX = (sumX + (w * sfX[f]));
        sumY = (sumY + (w * sfY[f]));
        sumZ = (sumZ + (w * sfZ[f]));
      }
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumX * invV);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumY * invV);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZ * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_grad_vol_vector_gauss = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_grad_vol_vector_gauss(const float* cellUx, const float* cellUy, const float* cellUz, const float* bcX, const float* bcY, const float* bcZ, const unsigned int* bcMask, const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* params, float* outXX, float* outXY, float* outXZ, float* outYX, float* outYY, float* outYZ, float* outZX, float* outZY, float* outZZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float sumXX = __uint_as_float(0x00000000u);
    float sumXY = __uint_as_float(0x00000000u);
    float sumXZ = __uint_as_float(0x00000000u);
    float sumYX = __uint_as_float(0x00000000u);
    float sumYY = __uint_as_float(0x00000000u);
    float sumYZ = __uint_as_float(0x00000000u);
    float sumZX = __uint_as_float(0x00000000u);
    float sumZY = __uint_as_float(0x00000000u);
    float sumZZ = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        float s = sign[k];
        int o = owner[f];
        float uxO = cellUx[o];
        float uyO = cellUy[o];
        float uzO = cellUz[o];
        float uxF = uxO;
        float uyF = uyO;
        float uzF = uzO;
        if (f < params[2]) {
          int n = neighbour[f];
          float uxN = cellUx[n];
          float uyN = cellUy[n];
          float uzN = cellUz[n];
          float ww = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - ww);
          uxF = ((ww * uxO) + (iw * uxN));
          uyF = ((ww * uyO) + (iw * uyN));
          uzF = ((ww * uzO) + (iw * uzN));
        } else {
          unsigned int m = bcMask[f];
          if (m == 1u) {
            float ww = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - ww);
            uxF = ((ww * uxO) + (iw * bcX[f]));
            uyF = ((ww * uyO) + (iw * bcY[f]));
            uzF = ((ww * uzO) + (iw * bcZ[f]));
          }
          if (m == 2u) {
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
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outXX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumXX * invV);
    outXY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumXY * invV);
    outXZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumXZ * invV);
    outYX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumYX * invV);
    outYY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumYY * invV);
    outYZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumYZ * invV);
    outZX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZX * invV);
    outZY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZY * invV);
    outZZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZZ * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_interp_scalar_face = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_interp_scalar_face(const int* owner, const int* neighbour, const float* weights, const float* cellPhi, const float* bcValue, const unsigned int* bcMask, const int* params, float* outPhiF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float po = cellPhi[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float pn = cellPhi[n];
      float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w);
      outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * po) + (iw * pn));
    } else {
      if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 1u) {
        float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w);
        outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * po) + (iw * bcValue[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      } else {
        if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 2u) {
          outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcValue[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        } else {
          outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = po;
        }
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_interp_scalar_face_all = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_interp_scalar_face_all(const float* cellPhi, const int* owner, const int* neighbour, const float* weights, const float* bcValue, const int* params, float* outPhiF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float po = cellPhi[o];
      float pn = cellPhi[n];
      float wf = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (wf < __uint_as_float(0x00000000u)) {
        wf = __uint_as_float(0x00000000u);
      }
      if (wf > __uint_as_float(0x3f800000u)) {
        wf = __uint_as_float(0x3f800000u);
      }
      float iw = (__uint_as_float(0x3f800000u) - wf);
      float a = ((wf * po) + (iw * pn));
      if (a < __uint_as_float(0x00000000u)) {
        a = __uint_as_float(0x00000000u);
      }
      if (a > __uint_as_float(0x3f800000u)) {
        a = __uint_as_float(0x3f800000u);
      }
      outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = a;
    } else {
      float a = bcValue[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (a < __uint_as_float(0x00000000u)) {
        a = __uint_as_float(0x00000000u);
      }
      if (a > __uint_as_float(0x3f800000u)) {
        a = __uint_as_float(0x3f800000u);
      }
      outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = a;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_interp_vector_face = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_interp_vector_face(const int* owner, const int* neighbour, const float* weights, const float* cellX, const float* cellY, const float* cellZ, const float* bcX, const float* bcY, const float* bcZ, const unsigned int* bcMask, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float uxo = cellX[o];
    float uyo = cellY[o];
    float uzo = cellZ[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float uxn = cellX[n];
      float uyn = cellY[n];
      float uzn = cellZ[n];
      float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w);
      outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uxo) + (iw * uxn));
      outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uyo) + (iw * uyn));
      outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uzo) + (iw * uzn));
    } else {
      unsigned int m = bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (m == 1u) {
        float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w);
        outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uxo) + (iw * bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uyo) + (iw * bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uzo) + (iw * bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      } else {
        if (m == 2u) {
          outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        } else {
          outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = uxo;
          outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = uyo;
          outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = uzo;
        }
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_linear_upwind_scalar_face = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_linear_upwind_scalar_face(const float* flux, const float* xCell, const float* gradX, const float* gradY, const float* gradZ, const float* limiter, const int* owner, const int* nei, const float* cx, const float* cy, const float* cz, const float* cfx, const float* cfy, const float* cfz, const float* bcValue, const unsigned int* bcMask, const unsigned int* params, float* outFace) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    float phif = flux[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int own = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) < ((int)(params[1]))) {
      if (phif >= __uint_as_float(0x00000000u)) {
        float dx = (cfx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cx[own]);
        float dy = (cfy[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cy[own]);
        float dz = (cfz[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cz[own]);
        float corr = (((dx * gradX[own]) + (dy * gradY[own])) + (dz * gradZ[own]));
        outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (xCell[own] + (limiter[own] * corr));
      } else {
        int neiCell = nei[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
        float dx = (cfx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cx[neiCell]);
        float dy = (cfy[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cy[neiCell]);
        float dz = (cfz[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cz[neiCell]);
        float corr = (((dx * gradX[neiCell]) + (dy * gradY[neiCell])) + (dz * gradZ[neiCell]));
        outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (xCell[neiCell] + (limiter[neiCell] * corr));
      }
    } else {
      if (phif >= __uint_as_float(0x00000000u)) {
        outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = xCell[own];
      } else {
        unsigned int mask = bcMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
        if (((int)(mask)) == 0) {
          outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = xCell[own];
        } else {
          outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = bcValue[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
        }
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_phi_from_u = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_phi_from_u(const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const float* ux, const float* uy, const float* uz, const float* bcX, const float* bcY, const float* bcZ, const unsigned int* bcMask, const int* params, float* outPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w);
      float ufx = ((w * uox) + (iw * ux[n]));
      float ufy = ((w * uoy) + (iw * uy[n]));
      float ufz = ((w * uoz) + (iw * uz[n]));
      outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufx) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufz));
    } else {
      if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 1u) {
        float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w);
        float ufx = ((w * uox) + (iw * bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        float ufy = ((w * uoy) + (iw * bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        float ufz = ((w * uoz) + (iw * bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufx) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufz));
      } else {
        if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 2u) {
          float ufx = bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          float ufy = bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          float ufz = bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufx) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufz));
        } else {
          outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * uox) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * uoy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * uoz));
        }
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_phi_from_u_boundary = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_phi_from_u_boundary(const int* owner, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const float* ux, const float* uy, const float* uz, const float* bcx, const float* bcy, const float* bcz, const unsigned int* bcmask, const unsigned int* params, float* outPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int f = (((int)(params[1])) + (int)(blockIdx.x * blockDim.x + threadIdx.x));
  if (f >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[f];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    float ufx = uox;
    float ufy = uoy;
    float ufz = uoz;
    unsigned int m = bcmask[f];
    int mInt = ((int)(m));
    if (mInt == 1) {
      float w = weights[f];
      float iw = (__uint_as_float(0x3f800000u) - w);
      ufx = ((w * uox) + (iw * bcx[f]));
      ufy = ((w * uoy) + (iw * bcy[f]));
      ufz = ((w * uoz) + (iw * bcz[f]));
    } else {
      if (mInt == 2) {
        ufx = bcx[f];
        ufy = bcy[f];
        ufz = bcz[f];
      }
    }
    outPhi[f] = (((sfX[f] * ufx) + (sfY[f] * ufy)) + (sfZ[f] * ufz));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_phi_from_u_internal = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_phi_from_u_internal(const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const float* ux, const float* uy, const float* uz, const unsigned int* params, float* outPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[1]))) {
    return;
  } else {
    int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int n = neighbour[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float w = weights[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float iw = (__uint_as_float(0x3f800000u) - w);
    float ufx = ((w * ux[o]) + (iw * ux[n]));
    float ufy = ((w * uy[o]) + (iw * uy[n]));
    float ufz = ((w * uz[o]) + (iw * uz[n]));
    outPhi[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (((sfX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ufx) + (sfY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ufy)) + (sfZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ufz));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_phig_gravity = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_phig_gravity(const float* ghf, const float* snGradRho, const float* rAUf, const float* magSf, const unsigned int* params, float* outPhig) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    outPhig[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (-((ghf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * snGradRho[(int)(blockIdx.x * blockDim.x + threadIdx.x)]) * (rAUf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * magSf[(int)(blockIdx.x * blockDim.x + threadIdx.x)])));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_dt_rho(const float* rho, const float* paramsF, const int* counts, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float r = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (paramsF[0] / r);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_dt_rho_integrated(const float* rho, const float* vol, const float* paramsF, const int* counts, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float r = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float denom = (r * v);
    if (denom != __uint_as_float(0x00000000u)) {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (paramsF[0] / denom);
    } else {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_dt_rho_sp(const float* rho, const float* sp, const float* paramsF, const int* counts, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float r = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float s = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float denom = (r + (paramsF[0] * s));
    outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (paramsF[0] / denom);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated(const float* rho, const float* sp, const float* vol, const float* paramsF, const int* counts, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float r = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float s = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float denom = ((r + (paramsF[0] * s)) * v);
    if (denom != __uint_as_float(0x00000000u)) {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (paramsF[0] / denom);
    } else {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_u_eqn_approx(const float* rhoCell, const float* muCell, const float* phiFaces, const float* rhoBcVal, const unsigned int* rhoBcMask, const float* muBcVal, const unsigned int* muBcMask, const int* ownerAll, const int* neighbourInt, const float* weightsAll, const float* magSfAll, const float* deltaAll, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const float* volCells, const int* counts3, const float* paramsF, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts3[0]) {
    return;
  } else {
    float invDt = (__uint_as_float(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float A = (rhoCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * invDt);
    float conv = __uint_as_float(0x00000000u);
    float diff = __uint_as_float(0x00000000u);
    unsigned int begU = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    unsigned int endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int fU = faceIdx[k];
      int f = ((int)(fU));
      if (f >= counts3[1]) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = __uint_as_float(0x00000000u);
        float muF = __uint_as_float(0x00000000u);
        if (f < counts3[2]) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (__uint_as_float(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if (rhoBcMask[f] != 0u) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if (muBcMask[f] != 0u) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if (s < 0) {
          outFlux = (__uint_as_float(0x00000000u) - rhoPhi);
        }
        if (outFlux > __uint_as_float(0x00000000u)) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    if (A != __uint_as_float(0x00000000u)) {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (__uint_as_float(0x3f800000u) / A);
    } else {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp(const float* rhoCell, const float* muCell, const float* phiFaces, const float* rhoBcVal, const unsigned int* rhoBcMask, const float* muBcVal, const unsigned int* muBcMask, const int* ownerAll, const int* neighbourInt, const float* weightsAll, const float* magSfAll, const float* deltaAll, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const float* volCells, const float* sp, const int* counts3, const float* paramsF, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts3[0]) {
    return;
  } else {
    float invDt = (__uint_as_float(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float A = (rhoCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * invDt);
    float conv = __uint_as_float(0x00000000u);
    float diff = __uint_as_float(0x00000000u);
    unsigned int begU = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    unsigned int endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int fU = faceIdx[k];
      int f = ((int)(fU));
      if (f >= counts3[1]) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = __uint_as_float(0x00000000u);
        float muF = __uint_as_float(0x00000000u);
        if (f < counts3[2]) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (__uint_as_float(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if (rhoBcMask[f] != 0u) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if (muBcMask[f] != 0u) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if (s < 0) {
          outFlux = (__uint_as_float(0x00000000u) - rhoPhi);
        }
        if (outFlux > __uint_as_float(0x00000000u)) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    A = (A + sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    if (A != __uint_as_float(0x00000000u)) {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (__uint_as_float(0x3f800000u) / A);
    } else {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_rho_from_alpha = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_rho_from_alpha(const float* alpha, const float* paramsF, const int* counts, float* outRho) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float a = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (a < __uint_as_float(0x00000000u)) {
      a = __uint_as_float(0x00000000u);
    }
    if (a > __uint_as_float(0x3f800000u)) {
      a = __uint_as_float(0x3f800000u);
    }
    float invA = (__uint_as_float(0x3f800000u) - a);
    float r = ((a * paramsF[0]) + (invA * paramsF[1]));
    outRho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = r;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_rho_phi_from_rho_and_phi = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_rho_phi_from_rho_and_phi(const float* phiAll, const float* rhoCell, const int* owner, const int* neighbour, const float* weights, const float* rhoBcVal, const unsigned int* rhoBcMask, const int* params, float* outRhoPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float rhoF = rhoCell[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float w0 = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w0);
      rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
    } else {
      unsigned int m = rhoBcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (m == 1u) {
        float w0 = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w0);
        rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      }
      if (m == 2u) {
        rhoF = rhoBcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      }
    }
    float flux = phiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outRhoPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (rhoF * flux);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_primitives_sn_grad_scalar = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_sn_grad_scalar(const float* cellVal, const int* owner, const int* neighbour, const float* delta, const float* faceVal, const unsigned int* faceMask, const int* params, float* outSnGrad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float aO = cellVal[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float aN = cellVal[n];
      outSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (aN - aO));
    } else {
      if (faceMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u) {
        float aN = faceVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        outSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (aN - aO));
      } else {
        outSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      }
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_average_face_scalar_to_cell[] = {
  { "facePhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumMag", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCell", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_average_face_scalar_to_cell = {
  1,
  "navatala_cfd_primitives_average_face_scalar_to_cell",
  "cuda",
  "navatala_cfd_primitives_average_face_scalar_to_cell",
  "kernel:cuda:navatala_cfd_primitives_average_face_scalar_to_cell",
  "abi-r1:cuda:navatala_cfd_primitives_average_face_scalar_to_cell",
  "abi-r1:cuda:navatala_cfd_primitives_average_face_scalar_to_cell",
  7,
  kAbiArgs_cuda_navatala_cfd_primitives_average_face_scalar_to_cell
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_evaluate_scalar_bc[] = {
  { "internalField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCells", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcTypeMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedValues", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "boundaryOut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_evaluate_scalar_bc = {
  1,
  "navatala_cfd_primitives_evaluate_scalar_bc",
  "cuda",
  "navatala_cfd_primitives_evaluate_scalar_bc",
  "kernel:cuda:navatala_cfd_primitives_evaluate_scalar_bc",
  "abi-r1:cuda:navatala_cfd_primitives_evaluate_scalar_bc",
  "abi-r1:cuda:navatala_cfd_primitives_evaluate_scalar_bc",
  6,
  kAbiArgs_cuda_navatala_cfd_primitives_evaluate_scalar_bc
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_evaluate_vector_bc[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_evaluate_vector_bc = {
  1,
  "navatala_cfd_primitives_evaluate_vector_bc",
  "cuda",
  "navatala_cfd_primitives_evaluate_vector_bc",
  "kernel:cuda:navatala_cfd_primitives_evaluate_vector_bc",
  "abi-r1:cuda:navatala_cfd_primitives_evaluate_vector_bc",
  "abi-r1:cuda:navatala_cfd_primitives_evaluate_vector_bc",
  12,
  kAbiArgs_cuda_navatala_cfd_primitives_evaluate_vector_bc
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_div_face_flux_csr[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFlux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiv", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_div_face_flux_csr = {
  1,
  "navatala_cfd_primitives_div_face_flux_csr",
  "cuda",
  "navatala_cfd_primitives_div_face_flux_csr",
  "kernel:cuda:navatala_cfd_primitives_div_face_flux_csr",
  "abi-r1:cuda:navatala_cfd_primitives_div_face_flux_csr",
  "abi-r1:cuda:navatala_cfd_primitives_div_face_flux_csr",
  7,
  kAbiArgs_cuda_navatala_cfd_primitives_div_face_flux_csr
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_div_face_flux_csr_f64[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFlux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiv", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_div_face_flux_csr_f64 = {
  1,
  "navatala_cfd_primitives_div_face_flux_csr_f64",
  "cuda",
  "navatala_cfd_primitives_div_face_flux_csr_f64",
  "kernel:cuda:navatala_cfd_primitives_div_face_flux_csr_f64",
  "abi-r1:cuda:navatala_cfd_primitives_div_face_flux_csr_f64",
  "abi-r1:cuda:navatala_cfd_primitives_div_face_flux_csr_f64",
  7,
  kAbiArgs_cuda_navatala_cfd_primitives_div_face_flux_csr_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_div_face_flux_vector_csr[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_div_face_flux_vector_csr = {
  1,
  "navatala_cfd_primitives_div_face_flux_vector_csr",
  "cuda",
  "navatala_cfd_primitives_div_face_flux_vector_csr",
  "kernel:cuda:navatala_cfd_primitives_div_face_flux_vector_csr",
  "abi-r1:cuda:navatala_cfd_primitives_div_face_flux_vector_csr",
  "abi-r1:cuda:navatala_cfd_primitives_div_face_flux_vector_csr",
  11,
  kAbiArgs_cuda_navatala_cfd_primitives_div_face_flux_vector_csr
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2 = {
  1,
  "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "cuda",
  "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "kernel:cuda:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "abi-r1:cuda:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "abi-r1:cuda:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  25,
  kAbiArgs_cuda_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_face_scalar_product[] = {
  { "aAllFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bAllFaces", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outProd", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_face_scalar_product = {
  1,
  "navatala_cfd_primitives_face_scalar_product",
  "cuda",
  "navatala_cfd_primitives_face_scalar_product",
  "kernel:cuda:navatala_cfd_primitives_face_scalar_product",
  "abi-r1:cuda:navatala_cfd_primitives_face_scalar_product",
  "abi-r1:cuda:navatala_cfd_primitives_face_scalar_product",
  4,
  kAbiArgs_cuda_navatala_cfd_primitives_face_scalar_product
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_grad_vol_scalar_gauss[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_grad_vol_scalar_gauss = {
  1,
  "navatala_cfd_primitives_grad_vol_scalar_gauss",
  "cuda",
  "navatala_cfd_primitives_grad_vol_scalar_gauss",
  "kernel:cuda:navatala_cfd_primitives_grad_vol_scalar_gauss",
  "abi-r1:cuda:navatala_cfd_primitives_grad_vol_scalar_gauss",
  "abi-r1:cuda:navatala_cfd_primitives_grad_vol_scalar_gauss",
  17,
  kAbiArgs_cuda_navatala_cfd_primitives_grad_vol_scalar_gauss
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_grad_vol_vector_gauss[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_grad_vol_vector_gauss = {
  1,
  "navatala_cfd_primitives_grad_vol_vector_gauss",
  "cuda",
  "navatala_cfd_primitives_grad_vol_vector_gauss",
  "kernel:cuda:navatala_cfd_primitives_grad_vol_vector_gauss",
  "abi-r1:cuda:navatala_cfd_primitives_grad_vol_vector_gauss",
  "abi-r1:cuda:navatala_cfd_primitives_grad_vol_vector_gauss",
  27,
  kAbiArgs_cuda_navatala_cfd_primitives_grad_vol_vector_gauss
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_interp_scalar_face[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellPhi", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhiF", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_interp_scalar_face = {
  1,
  "navatala_cfd_primitives_interp_scalar_face",
  "cuda",
  "navatala_cfd_primitives_interp_scalar_face",
  "kernel:cuda:navatala_cfd_primitives_interp_scalar_face",
  "abi-r1:cuda:navatala_cfd_primitives_interp_scalar_face",
  "abi-r1:cuda:navatala_cfd_primitives_interp_scalar_face",
  8,
  kAbiArgs_cuda_navatala_cfd_primitives_interp_scalar_face
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_interp_scalar_face_all[] = {
  { "cellPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhiF", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_interp_scalar_face_all = {
  1,
  "navatala_cfd_primitives_interp_scalar_face_all",
  "cuda",
  "navatala_cfd_primitives_interp_scalar_face_all",
  "kernel:cuda:navatala_cfd_primitives_interp_scalar_face_all",
  "abi-r1:cuda:navatala_cfd_primitives_interp_scalar_face_all",
  "abi-r1:cuda:navatala_cfd_primitives_interp_scalar_face_all",
  7,
  kAbiArgs_cuda_navatala_cfd_primitives_interp_scalar_face_all
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_interp_vector_face[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_interp_vector_face = {
  1,
  "navatala_cfd_primitives_interp_vector_face",
  "cuda",
  "navatala_cfd_primitives_interp_vector_face",
  "kernel:cuda:navatala_cfd_primitives_interp_vector_face",
  "abi-r1:cuda:navatala_cfd_primitives_interp_vector_face",
  "abi-r1:cuda:navatala_cfd_primitives_interp_vector_face",
  14,
  kAbiArgs_cuda_navatala_cfd_primitives_interp_vector_face
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_linear_upwind_scalar_face[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_linear_upwind_scalar_face = {
  1,
  "navatala_cfd_primitives_linear_upwind_scalar_face",
  "cuda",
  "navatala_cfd_primitives_linear_upwind_scalar_face",
  "kernel:cuda:navatala_cfd_primitives_linear_upwind_scalar_face",
  "abi-r1:cuda:navatala_cfd_primitives_linear_upwind_scalar_face",
  "abi-r1:cuda:navatala_cfd_primitives_linear_upwind_scalar_face",
  18,
  kAbiArgs_cuda_navatala_cfd_primitives_linear_upwind_scalar_face
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_phi_from_u[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_phi_from_u = {
  1,
  "navatala_cfd_primitives_phi_from_u",
  "cuda",
  "navatala_cfd_primitives_phi_from_u",
  "kernel:cuda:navatala_cfd_primitives_phi_from_u",
  "abi-r1:cuda:navatala_cfd_primitives_phi_from_u",
  "abi-r1:cuda:navatala_cfd_primitives_phi_from_u",
  15,
  kAbiArgs_cuda_navatala_cfd_primitives_phi_from_u
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_phi_from_u_boundary[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_phi_from_u_boundary = {
  1,
  "navatala_cfd_primitives_phi_from_u_boundary",
  "cuda",
  "navatala_cfd_primitives_phi_from_u_boundary",
  "kernel:cuda:navatala_cfd_primitives_phi_from_u_boundary",
  "abi-r1:cuda:navatala_cfd_primitives_phi_from_u_boundary",
  "abi-r1:cuda:navatala_cfd_primitives_phi_from_u_boundary",
  14,
  kAbiArgs_cuda_navatala_cfd_primitives_phi_from_u_boundary
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_phi_from_u_internal[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_phi_from_u_internal = {
  1,
  "navatala_cfd_primitives_phi_from_u_internal",
  "cuda",
  "navatala_cfd_primitives_phi_from_u_internal",
  "kernel:cuda:navatala_cfd_primitives_phi_from_u_internal",
  "abi-r1:cuda:navatala_cfd_primitives_phi_from_u_internal",
  "abi-r1:cuda:navatala_cfd_primitives_phi_from_u_internal",
  11,
  kAbiArgs_cuda_navatala_cfd_primitives_phi_from_u_internal
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_phig_gravity[] = {
  { "ghf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "snGradRho", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhig", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_phig_gravity = {
  1,
  "navatala_cfd_primitives_phig_gravity",
  "cuda",
  "navatala_cfd_primitives_phig_gravity",
  "kernel:cuda:navatala_cfd_primitives_phig_gravity",
  "abi-r1:cuda:navatala_cfd_primitives_phig_gravity",
  "abi-r1:cuda:navatala_cfd_primitives_phig_gravity",
  6,
  kAbiArgs_cuda_navatala_cfd_primitives_phig_gravity
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho",
  "cuda",
  "navatala_cfd_primitives_r_a_u_from_dt_rho",
  "kernel:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho",
  4,
  kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "cuda",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "kernel:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  5,
  kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "cuda",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "kernel:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  5,
  kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "cuda",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "kernel:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  6,
  kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx = {
  1,
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "cuda",
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "kernel:cuda:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  19,
  kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp = {
  1,
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "cuda",
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "kernel:cuda:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "abi-r1:cuda:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  20,
  kAbiArgs_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_rho_from_alpha[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRho", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_rho_from_alpha = {
  1,
  "navatala_cfd_primitives_rho_from_alpha",
  "cuda",
  "navatala_cfd_primitives_rho_from_alpha",
  "kernel:cuda:navatala_cfd_primitives_rho_from_alpha",
  "abi-r1:cuda:navatala_cfd_primitives_rho_from_alpha",
  "abi-r1:cuda:navatala_cfd_primitives_rho_from_alpha",
  4,
  kAbiArgs_cuda_navatala_cfd_primitives_rho_from_alpha
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_rho_phi_from_rho_and_phi[] = {
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
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_rho_phi_from_rho_and_phi = {
  1,
  "navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "cuda",
  "navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "kernel:cuda:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "abi-r1:cuda:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "abi-r1:cuda:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  9,
  kAbiArgs_cuda_navatala_cfd_primitives_rho_phi_from_rho_and_phi
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_primitives_sn_grad_scalar[] = {
  { "cellVal", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceVal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSnGrad", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_primitives_sn_grad_scalar = {
  1,
  "navatala_cfd_primitives_sn_grad_scalar",
  "cuda",
  "navatala_cfd_primitives_sn_grad_scalar",
  "kernel:cuda:navatala_cfd_primitives_sn_grad_scalar",
  "abi-r1:cuda:navatala_cfd_primitives_sn_grad_scalar",
  "abi-r1:cuda:navatala_cfd_primitives_sn_grad_scalar",
  8,
  kAbiArgs_cuda_navatala_cfd_primitives_sn_grad_scalar
};

bool tryGetKernelAbiManifest_cuda_cfd_primitives(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_average_face_scalar_to_cell") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_average_face_scalar_to_cell;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_evaluate_scalar_bc") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_evaluate_scalar_bc;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_evaluate_vector_bc") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_evaluate_vector_bc;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_face_flux_csr") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_div_face_flux_csr;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_face_flux_csr_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_div_face_flux_csr_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_face_flux_vector_csr") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_div_face_flux_vector_csr;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_face_scalar_product") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_face_scalar_product;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_grad_vol_scalar_gauss") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_grad_vol_scalar_gauss;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_grad_vol_vector_gauss") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_grad_vol_vector_gauss;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_interp_scalar_face") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_interp_scalar_face;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_interp_scalar_face_all") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_interp_scalar_face_all;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_interp_vector_face") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_interp_vector_face;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_linear_upwind_scalar_face") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_linear_upwind_scalar_face;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phi_from_u") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_phi_from_u;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phi_from_u_boundary") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_phi_from_u_boundary;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phi_from_u_internal") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_phi_from_u_internal;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phig_gravity") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_phig_gravity;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_rho_from_alpha") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_rho_from_alpha;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_rho_phi_from_rho_and_phi") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_rho_phi_from_rho_and_phi;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_sn_grad_scalar") {
    out = &kAbiManifest_cuda_navatala_cfd_primitives_sn_grad_scalar;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_cfd_primitives(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_average_face_scalar_to_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_average_face_scalar_to_cell";
    std::string_view sv(k_cuda_navatala_cfd_primitives_average_face_scalar_to_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_evaluate_scalar_bc") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_evaluate_scalar_bc";
    std::string_view sv(k_cuda_navatala_cfd_primitives_evaluate_scalar_bc);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_evaluate_vector_bc") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_evaluate_vector_bc";
    std::string_view sv(k_cuda_navatala_cfd_primitives_evaluate_vector_bc);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_face_flux_csr") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_csr";
    std::string_view sv(k_cuda_navatala_cfd_primitives_div_face_flux_csr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_face_flux_csr_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_csr_f64";
    std::string_view sv(k_cuda_navatala_cfd_primitives_div_face_flux_csr_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_face_flux_vector_csr") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_vector_csr";
    std::string_view sv(k_cuda_navatala_cfd_primitives_div_face_flux_vector_csr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2";
    std::string_view sv(k_cuda_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_face_scalar_product") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_face_scalar_product";
    std::string_view sv(k_cuda_navatala_cfd_primitives_face_scalar_product);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_grad_vol_scalar_gauss") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_grad_vol_scalar_gauss";
    std::string_view sv(k_cuda_navatala_cfd_primitives_grad_vol_scalar_gauss);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_grad_vol_vector_gauss") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_grad_vol_vector_gauss";
    std::string_view sv(k_cuda_navatala_cfd_primitives_grad_vol_vector_gauss);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_interp_scalar_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_interp_scalar_face";
    std::string_view sv(k_cuda_navatala_cfd_primitives_interp_scalar_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_interp_scalar_face_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_interp_scalar_face_all";
    std::string_view sv(k_cuda_navatala_cfd_primitives_interp_scalar_face_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_interp_vector_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_interp_vector_face";
    std::string_view sv(k_cuda_navatala_cfd_primitives_interp_vector_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_linear_upwind_scalar_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_linear_upwind_scalar_face";
    std::string_view sv(k_cuda_navatala_cfd_primitives_linear_upwind_scalar_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phi_from_u") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u";
    std::string_view sv(k_cuda_navatala_cfd_primitives_phi_from_u);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phi_from_u_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u_boundary";
    std::string_view sv(k_cuda_navatala_cfd_primitives_phi_from_u_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phi_from_u_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u_internal";
    std::string_view sv(k_cuda_navatala_cfd_primitives_phi_from_u_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_phig_gravity") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_phig_gravity";
    std::string_view sv(k_cuda_navatala_cfd_primitives_phig_gravity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho";
    std::string_view sv(k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated";
    std::string_view sv(k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_sp";
    std::string_view sv(k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated";
    std::string_view sv(k_cuda_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_u_eqn_approx";
    std::string_view sv(k_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp";
    std::string_view sv(k_cuda_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_rho_from_alpha") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_rho_from_alpha";
    std::string_view sv(k_cuda_navatala_cfd_primitives_rho_from_alpha);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_rho_phi_from_rho_and_phi") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_rho_phi_from_rho_and_phi";
    std::string_view sv(k_cuda_navatala_cfd_primitives_rho_phi_from_rho_and_phi);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_primitives_sn_grad_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_primitives_sn_grad_scalar";
    std::string_view sv(k_cuda_navatala_cfd_primitives_sn_grad_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

