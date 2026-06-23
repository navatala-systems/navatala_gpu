// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_cfd_primitives` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_cfd_primitives_average_face_scalar_to_cell = R"kernel(
__kernel void navatala_cfd_primitives_average_face_scalar_to_cell(__global const float* facePhi, __global const float* magSf, __global const uint* offsets, __global const uint* faceIdx, __global const float* sumMag, __global const int* params, __global float* outCell) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    uint begU = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    float acc = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      float term = (magSf[f] * facePhi[f]);
      acc = (acc + term);
    }
    float den = sumMag[((int)((int)(get_global_id(0))))];
    if ((den < as_float(0x1e3ce508u))) {
      den = as_float(0x1e3ce508u);
    }
    outCell[((int)((int)(get_global_id(0))))] = (acc / den);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_evaluate_scalar_bc = R"kernel(
__kernel void navatala_cfd_primitives_evaluate_scalar_bc(__global const float* internalField, __global const int* faceCells, __global const uint* bcTypeMask, __global const float* fixedValues, __global const uint* counts, __global float* boundaryOut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    uint bcType = bcTypeMask[(int)(get_global_id(0))];
    int cell = faceCells[(int)(get_global_id(0))];
    if ((((int)(bcType)) == 1)) {
      boundaryOut[(int)(get_global_id(0))] = fixedValues[(int)(get_global_id(0))];
    } else {
      boundaryOut[(int)(get_global_id(0))] = internalField[cell];
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_evaluate_vector_bc = R"kernel(
__kernel void navatala_cfd_primitives_evaluate_vector_bc(__global const float* inX, __global const float* inY, __global const float* inZ, __global const int* faceCells, __global const uint* bcTypeMask, __global const float* fvX, __global const float* fvY, __global const float* fvZ, __global const uint* counts, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    uint bcType = bcTypeMask[(int)(get_global_id(0))];
    int cell = faceCells[(int)(get_global_id(0))];
    if ((((int)(bcType)) == 1)) {
      outX[(int)(get_global_id(0))] = fvX[(int)(get_global_id(0))];
      outY[(int)(get_global_id(0))] = fvY[(int)(get_global_id(0))];
      outZ[(int)(get_global_id(0))] = fvZ[(int)(get_global_id(0))];
    } else {
      outX[(int)(get_global_id(0))] = inX[cell];
      outY[(int)(get_global_id(0))] = inY[cell];
      outZ[(int)(get_global_id(0))] = inZ[cell];
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_div_face_flux_csr = R"kernel(
__kernel void navatala_cfd_primitives_div_face_flux_csr(__global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* faceFlux, __global const float* vol, __global const int* params, __global float* outDiv) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float acc = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float term = (sign[k] * faceFlux[f]);
      acc = (acc + term);
    }
    outDiv[((int)((int)(get_global_id(0))))] = (acc / vol[((int)((int)(get_global_id(0))))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_div_face_flux_csr_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_primitives_div_face_flux_csr_f64(__global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const double* faceFlux, __global const double* vol, __global const int* params, __global double* outDiv) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double acc = as_double(0x0000000000000000ul);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double term = (signV * faceFlux[f]);
      acc = (acc + term);
    }
    outDiv[((int)((int)(get_global_id(0))))] = (acc / ((double)(vol[((int)((int)(get_global_id(0))))])));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_div_face_flux_vector_csr = R"kernel(
__kernel void navatala_cfd_primitives_div_face_flux_vector_csr(__global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* faceFluxX, __global const float* faceFluxY, __global const float* faceFluxZ, __global const float* vol, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float accX = as_float(0x00000000u);
    float accY = as_float(0x00000000u);
    float accZ = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      accX = (accX + (s * faceFluxX[f]));
      accY = (accY + (s * faceFluxY[f]));
      accZ = (accZ + (s * faceFluxZ[f]));
    }
    float volCell = vol[((int)((int)(get_global_id(0))))];
    float volSafe = (((fabs(volCell) > as_float(0x1e3ce508u))) ? (volCell) : (as_float(0x1e3ce508u)));
    float invV = (as_float(0x3f800000u) / volSafe);
    outX[((int)((int)(get_global_id(0))))] = (accX * invV);
    outY[((int)((int)(get_global_id(0))))] = (accY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (accZ * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2 = R"kernel(
__kernel void navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2(__global const float* muCell, __global const float* muBc, __global const float* gXX, __global const float* gXY, __global const float* gXZ, __global const float* gYX, __global const float* gYY, __global const float* gYZ, __global const float* gZX, __global const float* gZY, __global const float* gZZ, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* signF, __global const float* vol, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float sumX = as_float(0x00000000u);
    float sumY = as_float(0x00000000u);
    float sumZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
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
        float twoThirdO = (as_float(0x3f2aaaabu) * trO);
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
          float twoThirdN = (as_float(0x3f2aaaabu) * trN);
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
          float iw = (as_float(0x3f800000u) - ww);
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
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if ((v != as_float(0x00000000u))) {
      invV = (as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(get_global_id(0))))] = (sumX * invV);
    outY[((int)((int)(get_global_id(0))))] = (sumY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (sumZ * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_face_scalar_product = R"kernel(
__kernel void navatala_cfd_primitives_face_scalar_product(__global const float* aAllFaces, __global const float* bAllFaces, __global const int* params, __global float* outProd) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float a = aAllFaces[((int)((int)(get_global_id(0))))];
    float b = bAllFaces[((int)((int)(get_global_id(0))))];
    outProd[((int)((int)(get_global_id(0))))] = (a * b);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_grad_vol_scalar_gauss = R"kernel(
__kernel void navatala_cfd_primitives_grad_vol_scalar_gauss(__global const float* cellPhi, __global const float* bcValue, __global const uint* bcMask, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float sumX = as_float(0x00000000u);
    float sumY = as_float(0x00000000u);
    float sumZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
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
          float iw = (as_float(0x3f800000u) - ww);
          phiF = ((ww * po) + (iw * pn));
        } else {
          uint m = bcMask[f];
          if ((m == (uint)(1u))) {
            float ww = weights[f];
            float iw = (as_float(0x3f800000u) - ww);
            phiF = ((ww * po) + (iw * bcValue[f]));
          }
          if ((m == (uint)(2u))) {
            phiF = bcValue[f];
          }
        }
        float w = (s * phiF);
        sumX = (sumX + (w * sfX[f]));
        sumY = (sumY + (w * sfY[f]));
        sumZ = (sumZ + (w * sfZ[f]));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if ((v != as_float(0x00000000u))) {
      invV = (as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(get_global_id(0))))] = (sumX * invV);
    outY[((int)((int)(get_global_id(0))))] = (sumY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (sumZ * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_grad_vol_vector_gauss = R"kernel(
__kernel void navatala_cfd_primitives_grad_vol_vector_gauss(__global const float* cellUx, __global const float* cellUy, __global const float* cellUz, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const uint* bcMask, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* params, __global float* outXX, __global float* outXY, __global float* outXZ, __global float* outYX, __global float* outYY, __global float* outYZ, __global float* outZX, __global float* outZY, __global float* outZZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float sumXX = as_float(0x00000000u);
    float sumXY = as_float(0x00000000u);
    float sumXZ = as_float(0x00000000u);
    float sumYX = as_float(0x00000000u);
    float sumYY = as_float(0x00000000u);
    float sumYZ = as_float(0x00000000u);
    float sumZX = as_float(0x00000000u);
    float sumZY = as_float(0x00000000u);
    float sumZZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
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
          float iw = (as_float(0x3f800000u) - ww);
          uxF = ((ww * uxO) + (iw * uxN));
          uyF = ((ww * uyO) + (iw * uyN));
          uzF = ((ww * uzO) + (iw * uzN));
        } else {
          uint m = bcMask[f];
          if ((m == (uint)(1u))) {
            float ww = weights[f];
            float iw = (as_float(0x3f800000u) - ww);
            uxF = ((ww * uxO) + (iw * bcX[f]));
            uyF = ((ww * uyO) + (iw * bcY[f]));
            uzF = ((ww * uzO) + (iw * bcZ[f]));
          }
          if ((m == (uint)(2u))) {
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
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if ((v != as_float(0x00000000u))) {
      invV = (as_float(0x3f800000u) / v);
    }
    outXX[((int)((int)(get_global_id(0))))] = (sumXX * invV);
    outXY[((int)((int)(get_global_id(0))))] = (sumXY * invV);
    outXZ[((int)((int)(get_global_id(0))))] = (sumXZ * invV);
    outYX[((int)((int)(get_global_id(0))))] = (sumYX * invV);
    outYY[((int)((int)(get_global_id(0))))] = (sumYY * invV);
    outYZ[((int)((int)(get_global_id(0))))] = (sumYZ * invV);
    outZX[((int)((int)(get_global_id(0))))] = (sumZX * invV);
    outZY[((int)((int)(get_global_id(0))))] = (sumZY * invV);
    outZZ[((int)((int)(get_global_id(0))))] = (sumZZ * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_interp_scalar_face = R"kernel(
__kernel void navatala_cfd_primitives_interp_scalar_face(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* cellPhi, __global const float* bcValue, __global const uint* bcMask, __global const int* params, __global float* outPhiF) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float po = cellPhi[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float pn = cellPhi[n];
      float w = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w);
      outPhiF[((int)((int)(get_global_id(0))))] = ((w * po) + (iw * pn));
    } else {
      if ((bcMask[((int)((int)(get_global_id(0))))] == (uint)(1u))) {
        float w = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w);
        outPhiF[((int)((int)(get_global_id(0))))] = ((w * po) + (iw * bcValue[((int)((int)(get_global_id(0))))]));
      } else {
        if ((bcMask[((int)((int)(get_global_id(0))))] == (uint)(2u))) {
          outPhiF[((int)((int)(get_global_id(0))))] = bcValue[((int)((int)(get_global_id(0))))];
        } else {
          outPhiF[((int)((int)(get_global_id(0))))] = po;
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_interp_scalar_face_all = R"kernel(
__kernel void navatala_cfd_primitives_interp_scalar_face_all(__global const float* cellPhi, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* bcValue, __global const int* params, __global float* outPhiF) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int o = owner[((int)((int)(get_global_id(0))))];
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float po = cellPhi[o];
      float pn = cellPhi[n];
      float wf = weights[((int)((int)(get_global_id(0))))];
      if ((wf < as_float(0x00000000u))) {
        wf = as_float(0x00000000u);
      }
      if ((wf > as_float(0x3f800000u))) {
        wf = as_float(0x3f800000u);
      }
      float iw = (as_float(0x3f800000u) - wf);
      float a = ((wf * po) + (iw * pn));
      if ((a < as_float(0x00000000u))) {
        a = as_float(0x00000000u);
      }
      if ((a > as_float(0x3f800000u))) {
        a = as_float(0x3f800000u);
      }
      outPhiF[((int)((int)(get_global_id(0))))] = a;
    } else {
      float a = bcValue[((int)((int)(get_global_id(0))))];
      if ((a < as_float(0x00000000u))) {
        a = as_float(0x00000000u);
      }
      if ((a > as_float(0x3f800000u))) {
        a = as_float(0x3f800000u);
      }
      outPhiF[((int)((int)(get_global_id(0))))] = a;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_interp_vector_face = R"kernel(
__kernel void navatala_cfd_primitives_interp_vector_face(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* cellX, __global const float* cellY, __global const float* cellZ, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const uint* bcMask, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float uxo = cellX[o];
    float uyo = cellY[o];
    float uzo = cellZ[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float uxn = cellX[n];
      float uyn = cellY[n];
      float uzn = cellZ[n];
      float w = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w);
      outX[((int)((int)(get_global_id(0))))] = ((w * uxo) + (iw * uxn));
      outY[((int)((int)(get_global_id(0))))] = ((w * uyo) + (iw * uyn));
      outZ[((int)((int)(get_global_id(0))))] = ((w * uzo) + (iw * uzn));
    } else {
      uint m = bcMask[((int)((int)(get_global_id(0))))];
      if ((m == (uint)(1u))) {
        float w = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w);
        outX[((int)((int)(get_global_id(0))))] = ((w * uxo) + (iw * bcX[((int)((int)(get_global_id(0))))]));
        outY[((int)((int)(get_global_id(0))))] = ((w * uyo) + (iw * bcY[((int)((int)(get_global_id(0))))]));
        outZ[((int)((int)(get_global_id(0))))] = ((w * uzo) + (iw * bcZ[((int)((int)(get_global_id(0))))]));
      } else {
        if ((m == (uint)(2u))) {
          outX[((int)((int)(get_global_id(0))))] = bcX[((int)((int)(get_global_id(0))))];
          outY[((int)((int)(get_global_id(0))))] = bcY[((int)((int)(get_global_id(0))))];
          outZ[((int)((int)(get_global_id(0))))] = bcZ[((int)((int)(get_global_id(0))))];
        } else {
          outX[((int)((int)(get_global_id(0))))] = uxo;
          outY[((int)((int)(get_global_id(0))))] = uyo;
          outZ[((int)((int)(get_global_id(0))))] = uzo;
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_linear_upwind_scalar_face = R"kernel(
__kernel void navatala_cfd_primitives_linear_upwind_scalar_face(__global const float* flux, __global const float* xCell, __global const float* gradX, __global const float* gradY, __global const float* gradZ, __global const float* limiter, __global const int* owner, __global const int* nei, __global const float* cx, __global const float* cy, __global const float* cz, __global const float* cfx, __global const float* cfy, __global const float* cfz, __global const float* bcValue, __global const uint* bcMask, __global const uint* params, __global float* outFace) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    float phif = flux[(int)(get_global_id(0))];
    int own = owner[(int)(get_global_id(0))];
    if (((int)(get_global_id(0)) < ((int)(params[1])))) {
      if ((phif >= as_float(0x00000000u))) {
        float dx = (cfx[(int)(get_global_id(0))] - cx[own]);
        float dy = (cfy[(int)(get_global_id(0))] - cy[own]);
        float dz = (cfz[(int)(get_global_id(0))] - cz[own]);
        float corr = (((dx * gradX[own]) + (dy * gradY[own])) + (dz * gradZ[own]));
        outFace[(int)(get_global_id(0))] = (xCell[own] + (limiter[own] * corr));
      } else {
        int neiCell = nei[(int)(get_global_id(0))];
        float dx = (cfx[(int)(get_global_id(0))] - cx[neiCell]);
        float dy = (cfy[(int)(get_global_id(0))] - cy[neiCell]);
        float dz = (cfz[(int)(get_global_id(0))] - cz[neiCell]);
        float corr = (((dx * gradX[neiCell]) + (dy * gradY[neiCell])) + (dz * gradZ[neiCell]));
        outFace[(int)(get_global_id(0))] = (xCell[neiCell] + (limiter[neiCell] * corr));
      }
    } else {
      if ((phif >= as_float(0x00000000u))) {
        outFace[(int)(get_global_id(0))] = xCell[own];
      } else {
        uint mask = bcMask[(int)(get_global_id(0))];
        if ((((int)(mask)) == 0)) {
          outFace[(int)(get_global_id(0))] = xCell[own];
        } else {
          outFace[(int)(get_global_id(0))] = bcValue[(int)(get_global_id(0))];
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_phi_from_u = R"kernel(
__kernel void navatala_cfd_primitives_phi_from_u(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const uint* bcMask, __global const int* params, __global float* outPhi) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float w = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w);
      float ufx = ((w * uox) + (iw * ux[n]));
      float ufy = ((w * uoy) + (iw * uy[n]));
      float ufz = ((w * uoz) + (iw * uz[n]));
      outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * ufx) + (sfY[((int)((int)(get_global_id(0))))] * ufy)) + (sfZ[((int)((int)(get_global_id(0))))] * ufz));
    } else {
      if ((bcMask[((int)((int)(get_global_id(0))))] == (uint)(1u))) {
        float w = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w);
        float ufx = ((w * uox) + (iw * bcX[((int)((int)(get_global_id(0))))]));
        float ufy = ((w * uoy) + (iw * bcY[((int)((int)(get_global_id(0))))]));
        float ufz = ((w * uoz) + (iw * bcZ[((int)((int)(get_global_id(0))))]));
        outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * ufx) + (sfY[((int)((int)(get_global_id(0))))] * ufy)) + (sfZ[((int)((int)(get_global_id(0))))] * ufz));
      } else {
        if ((bcMask[((int)((int)(get_global_id(0))))] == (uint)(2u))) {
          float ufx = bcX[((int)((int)(get_global_id(0))))];
          float ufy = bcY[((int)((int)(get_global_id(0))))];
          float ufz = bcZ[((int)((int)(get_global_id(0))))];
          outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * ufx) + (sfY[((int)((int)(get_global_id(0))))] * ufy)) + (sfZ[((int)((int)(get_global_id(0))))] * ufz));
        } else {
          outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * uox) + (sfY[((int)((int)(get_global_id(0))))] * uoy)) + (sfZ[((int)((int)(get_global_id(0))))] * uoz));
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_phi_from_u_boundary = R"kernel(
__kernel void navatala_cfd_primitives_phi_from_u_boundary(__global const int* owner, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* bcx, __global const float* bcy, __global const float* bcz, __global const uint* bcmask, __global const uint* params, __global float* outPhi) {
  int gid0 = (int)get_global_id(0);
  int f = (((int)(params[1])) + (int)(get_global_id(0)));
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
      float iw = (as_float(0x3f800000u) - w);
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
const char* k_opencl_navatala_cfd_primitives_phi_from_u_internal = R"kernel(
__kernel void navatala_cfd_primitives_phi_from_u_internal(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* ux, __global const float* uy, __global const float* uz, __global const uint* params, __global float* outPhi) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[1])))) {
    return;
  } else {
    int o = owner[(int)(get_global_id(0))];
    int n = neighbour[(int)(get_global_id(0))];
    float w = weights[(int)(get_global_id(0))];
    float iw = (as_float(0x3f800000u) - w);
    float ufx = ((w * ux[o]) + (iw * ux[n]));
    float ufy = ((w * uy[o]) + (iw * uy[n]));
    float ufz = ((w * uz[o]) + (iw * uz[n]));
    outPhi[(int)(get_global_id(0))] = (((sfX[(int)(get_global_id(0))] * ufx) + (sfY[(int)(get_global_id(0))] * ufy)) + (sfZ[(int)(get_global_id(0))] * ufz));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_phig_gravity = R"kernel(
__kernel void navatala_cfd_primitives_phig_gravity(__global const float* ghf, __global const float* snGradRho, __global const float* rAUf, __global const float* magSf, __global const uint* params, __global float* outPhig) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    outPhig[(int)(get_global_id(0))] = (-((ghf[(int)(get_global_id(0))] * snGradRho[(int)(get_global_id(0))]) * (rAUf[(int)(get_global_id(0))] * magSf[(int)(get_global_id(0))])));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho = R"kernel(
__kernel void navatala_cfd_primitives_r_a_u_from_dt_rho(__global const float* rho, __global const float* paramsF, __global const int* counts, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)((int)(get_global_id(0))))];
    outRAU[((int)((int)(get_global_id(0))))] = (paramsF[0] / r);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated = R"kernel(
__kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_integrated(__global const float* rho, __global const float* vol, __global const float* paramsF, __global const int* counts, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)((int)(get_global_id(0))))];
    float v = vol[((int)((int)(get_global_id(0))))];
    float denom = (r * v);
    if ((denom != as_float(0x00000000u))) {
      outRAU[((int)((int)(get_global_id(0))))] = (paramsF[0] / denom);
    } else {
      outRAU[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp = R"kernel(
__kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_sp(__global const float* rho, __global const float* sp, __global const float* paramsF, __global const int* counts, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)((int)(get_global_id(0))))];
    float s = sp[((int)((int)(get_global_id(0))))];
    float denom = (r + (paramsF[0] * s));
    outRAU[((int)((int)(get_global_id(0))))] = (paramsF[0] / denom);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated = R"kernel(
__kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated(__global const float* rho, __global const float* sp, __global const float* vol, __global const float* paramsF, __global const int* counts, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)((int)(get_global_id(0))))];
    float s = sp[((int)((int)(get_global_id(0))))];
    float v = vol[((int)((int)(get_global_id(0))))];
    float denom = ((r + (paramsF[0] * s)) * v);
    if ((denom != as_float(0x00000000u))) {
      outRAU[((int)((int)(get_global_id(0))))] = (paramsF[0] / denom);
    } else {
      outRAU[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx = R"kernel(
__kernel void navatala_cfd_primitives_r_a_u_from_u_eqn_approx(__global const float* rhoCell, __global const float* muCell, __global const float* phiFaces, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const float* muBcVal, __global const uint* muBcMask, __global const int* ownerAll, __global const int* neighbourInt, __global const float* weightsAll, __global const float* magSfAll, __global const float* deltaAll, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const float* volCells, __global const int* counts3, __global const float* paramsF, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= counts3[0])) {
    return;
  } else {
    float invDt = (as_float(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)((int)(get_global_id(0))))];
    float A = (rhoCell[((int)((int)(get_global_id(0))))] * invDt);
    float conv = as_float(0x00000000u);
    float diff = as_float(0x00000000u);
    uint begU = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
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
        float rhoF = as_float(0x00000000u);
        float muF = as_float(0x00000000u);
        if ((f < counts3[2])) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (as_float(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if ((rhoBcMask[f] != (uint)(0u))) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if ((muBcMask[f] != (uint)(0u))) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if ((s < 0)) {
          outFlux = (as_float(0x00000000u) - rhoPhi);
        }
        if ((outFlux > as_float(0x00000000u))) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    if ((A != as_float(0x00000000u))) {
      outRAU[((int)((int)(get_global_id(0))))] = (as_float(0x3f800000u) / A);
    } else {
      outRAU[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp = R"kernel(
__kernel void navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp(__global const float* rhoCell, __global const float* muCell, __global const float* phiFaces, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const float* muBcVal, __global const uint* muBcMask, __global const int* ownerAll, __global const int* neighbourInt, __global const float* weightsAll, __global const float* magSfAll, __global const float* deltaAll, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const float* volCells, __global const float* sp, __global const int* counts3, __global const float* paramsF, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= counts3[0])) {
    return;
  } else {
    float invDt = (as_float(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)((int)(get_global_id(0))))];
    float A = (rhoCell[((int)((int)(get_global_id(0))))] * invDt);
    float conv = as_float(0x00000000u);
    float diff = as_float(0x00000000u);
    uint begU = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
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
        float rhoF = as_float(0x00000000u);
        float muF = as_float(0x00000000u);
        if ((f < counts3[2])) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (as_float(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if ((rhoBcMask[f] != (uint)(0u))) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if ((muBcMask[f] != (uint)(0u))) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if ((s < 0)) {
          outFlux = (as_float(0x00000000u) - rhoPhi);
        }
        if ((outFlux > as_float(0x00000000u))) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    A = (A + sp[((int)((int)(get_global_id(0))))]);
    if ((A != as_float(0x00000000u))) {
      outRAU[((int)((int)(get_global_id(0))))] = (as_float(0x3f800000u) / A);
    } else {
      outRAU[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_rho_from_alpha = R"kernel(
__kernel void navatala_cfd_primitives_rho_from_alpha(__global const float* alpha, __global const float* paramsF, __global const int* counts, __global float* outRho) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float a = alpha[((int)((int)(get_global_id(0))))];
    if ((a < as_float(0x00000000u))) {
      a = as_float(0x00000000u);
    }
    if ((a > as_float(0x3f800000u))) {
      a = as_float(0x3f800000u);
    }
    float invA = (as_float(0x3f800000u) - a);
    float r = ((a * paramsF[0]) + (invA * paramsF[1]));
    outRho[((int)((int)(get_global_id(0))))] = r;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_rho_phi_from_rho_and_phi = R"kernel(
__kernel void navatala_cfd_primitives_rho_phi_from_rho_and_phi(__global const float* phiAll, __global const float* rhoCell, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const int* params, __global float* outRhoPhi) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float rhoF = rhoCell[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float w0 = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w0);
      rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
    } else {
      uint m = rhoBcMask[((int)((int)(get_global_id(0))))];
      if ((m == (uint)(1u))) {
        float w0 = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w0);
        rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[((int)((int)(get_global_id(0))))]));
      }
      if ((m == (uint)(2u))) {
        rhoF = rhoBcVal[((int)((int)(get_global_id(0))))];
      }
    }
    float flux = phiAll[((int)((int)(get_global_id(0))))];
    outRhoPhi[((int)((int)(get_global_id(0))))] = (rhoF * flux);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_primitives_sn_grad_scalar = R"kernel(
__kernel void navatala_cfd_primitives_sn_grad_scalar(__global const float* cellVal, __global const int* owner, __global const int* neighbour, __global const float* delta, __global const float* faceVal, __global const uint* faceMask, __global const int* params, __global float* outSnGrad) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float aO = cellVal[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float aN = cellVal[n];
      outSnGrad[((int)((int)(get_global_id(0))))] = (delta[((int)((int)(get_global_id(0))))] * (aN - aO));
    } else {
      if ((faceMask[((int)((int)(get_global_id(0))))] != (uint)(0u))) {
        float aN = faceVal[((int)((int)(get_global_id(0))))];
        outSnGrad[((int)((int)(get_global_id(0))))] = (delta[((int)((int)(get_global_id(0))))] * (aN - aO));
      } else {
        outSnGrad[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      }
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_average_face_scalar_to_cell[] = {
  { "facePhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumMag", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCell", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_average_face_scalar_to_cell = {
  1,
  "navatala_cfd_primitives_average_face_scalar_to_cell",
  "opencl",
  "navatala_cfd_primitives_average_face_scalar_to_cell",
  "kernel:opencl:navatala_cfd_primitives_average_face_scalar_to_cell",
  "abi-r1:opencl:navatala_cfd_primitives_average_face_scalar_to_cell",
  "abi-r1:opencl:navatala_cfd_primitives_average_face_scalar_to_cell",
  7,
  kAbiArgs_opencl_navatala_cfd_primitives_average_face_scalar_to_cell
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_evaluate_scalar_bc[] = {
  { "internalField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceCells", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcTypeMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedValues", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "boundaryOut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_evaluate_scalar_bc = {
  1,
  "navatala_cfd_primitives_evaluate_scalar_bc",
  "opencl",
  "navatala_cfd_primitives_evaluate_scalar_bc",
  "kernel:opencl:navatala_cfd_primitives_evaluate_scalar_bc",
  "abi-r1:opencl:navatala_cfd_primitives_evaluate_scalar_bc",
  "abi-r1:opencl:navatala_cfd_primitives_evaluate_scalar_bc",
  6,
  kAbiArgs_opencl_navatala_cfd_primitives_evaluate_scalar_bc
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_evaluate_vector_bc[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_evaluate_vector_bc = {
  1,
  "navatala_cfd_primitives_evaluate_vector_bc",
  "opencl",
  "navatala_cfd_primitives_evaluate_vector_bc",
  "kernel:opencl:navatala_cfd_primitives_evaluate_vector_bc",
  "abi-r1:opencl:navatala_cfd_primitives_evaluate_vector_bc",
  "abi-r1:opencl:navatala_cfd_primitives_evaluate_vector_bc",
  12,
  kAbiArgs_opencl_navatala_cfd_primitives_evaluate_vector_bc
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_div_face_flux_csr[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFlux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiv", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_div_face_flux_csr = {
  1,
  "navatala_cfd_primitives_div_face_flux_csr",
  "opencl",
  "navatala_cfd_primitives_div_face_flux_csr",
  "kernel:opencl:navatala_cfd_primitives_div_face_flux_csr",
  "abi-r1:opencl:navatala_cfd_primitives_div_face_flux_csr",
  "abi-r1:opencl:navatala_cfd_primitives_div_face_flux_csr",
  7,
  kAbiArgs_opencl_navatala_cfd_primitives_div_face_flux_csr
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_div_face_flux_csr_f64[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFlux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiv", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_div_face_flux_csr_f64 = {
  1,
  "navatala_cfd_primitives_div_face_flux_csr_f64",
  "opencl",
  "navatala_cfd_primitives_div_face_flux_csr_f64",
  "kernel:opencl:navatala_cfd_primitives_div_face_flux_csr_f64",
  "abi-r1:opencl:navatala_cfd_primitives_div_face_flux_csr_f64",
  "abi-r1:opencl:navatala_cfd_primitives_div_face_flux_csr_f64",
  7,
  kAbiArgs_opencl_navatala_cfd_primitives_div_face_flux_csr_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_div_face_flux_vector_csr[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_div_face_flux_vector_csr = {
  1,
  "navatala_cfd_primitives_div_face_flux_vector_csr",
  "opencl",
  "navatala_cfd_primitives_div_face_flux_vector_csr",
  "kernel:opencl:navatala_cfd_primitives_div_face_flux_vector_csr",
  "abi-r1:opencl:navatala_cfd_primitives_div_face_flux_vector_csr",
  "abi-r1:opencl:navatala_cfd_primitives_div_face_flux_vector_csr",
  11,
  kAbiArgs_opencl_navatala_cfd_primitives_div_face_flux_vector_csr
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2 = {
  1,
  "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "opencl",
  "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "kernel:opencl:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "abi-r1:opencl:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  "abi-r1:opencl:navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2",
  25,
  kAbiArgs_opencl_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_face_scalar_product[] = {
  { "aAllFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bAllFaces", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outProd", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_face_scalar_product = {
  1,
  "navatala_cfd_primitives_face_scalar_product",
  "opencl",
  "navatala_cfd_primitives_face_scalar_product",
  "kernel:opencl:navatala_cfd_primitives_face_scalar_product",
  "abi-r1:opencl:navatala_cfd_primitives_face_scalar_product",
  "abi-r1:opencl:navatala_cfd_primitives_face_scalar_product",
  4,
  kAbiArgs_opencl_navatala_cfd_primitives_face_scalar_product
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_grad_vol_scalar_gauss[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_grad_vol_scalar_gauss = {
  1,
  "navatala_cfd_primitives_grad_vol_scalar_gauss",
  "opencl",
  "navatala_cfd_primitives_grad_vol_scalar_gauss",
  "kernel:opencl:navatala_cfd_primitives_grad_vol_scalar_gauss",
  "abi-r1:opencl:navatala_cfd_primitives_grad_vol_scalar_gauss",
  "abi-r1:opencl:navatala_cfd_primitives_grad_vol_scalar_gauss",
  17,
  kAbiArgs_opencl_navatala_cfd_primitives_grad_vol_scalar_gauss
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_grad_vol_vector_gauss[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_grad_vol_vector_gauss = {
  1,
  "navatala_cfd_primitives_grad_vol_vector_gauss",
  "opencl",
  "navatala_cfd_primitives_grad_vol_vector_gauss",
  "kernel:opencl:navatala_cfd_primitives_grad_vol_vector_gauss",
  "abi-r1:opencl:navatala_cfd_primitives_grad_vol_vector_gauss",
  "abi-r1:opencl:navatala_cfd_primitives_grad_vol_vector_gauss",
  27,
  kAbiArgs_opencl_navatala_cfd_primitives_grad_vol_vector_gauss
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_interp_scalar_face[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cellPhi", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhiF", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_interp_scalar_face = {
  1,
  "navatala_cfd_primitives_interp_scalar_face",
  "opencl",
  "navatala_cfd_primitives_interp_scalar_face",
  "kernel:opencl:navatala_cfd_primitives_interp_scalar_face",
  "abi-r1:opencl:navatala_cfd_primitives_interp_scalar_face",
  "abi-r1:opencl:navatala_cfd_primitives_interp_scalar_face",
  8,
  kAbiArgs_opencl_navatala_cfd_primitives_interp_scalar_face
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_interp_scalar_face_all[] = {
  { "cellPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcValue", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPhiF", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_interp_scalar_face_all = {
  1,
  "navatala_cfd_primitives_interp_scalar_face_all",
  "opencl",
  "navatala_cfd_primitives_interp_scalar_face_all",
  "kernel:opencl:navatala_cfd_primitives_interp_scalar_face_all",
  "abi-r1:opencl:navatala_cfd_primitives_interp_scalar_face_all",
  "abi-r1:opencl:navatala_cfd_primitives_interp_scalar_face_all",
  7,
  kAbiArgs_opencl_navatala_cfd_primitives_interp_scalar_face_all
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_interp_vector_face[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_interp_vector_face = {
  1,
  "navatala_cfd_primitives_interp_vector_face",
  "opencl",
  "navatala_cfd_primitives_interp_vector_face",
  "kernel:opencl:navatala_cfd_primitives_interp_vector_face",
  "abi-r1:opencl:navatala_cfd_primitives_interp_vector_face",
  "abi-r1:opencl:navatala_cfd_primitives_interp_vector_face",
  14,
  kAbiArgs_opencl_navatala_cfd_primitives_interp_vector_face
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_linear_upwind_scalar_face[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_linear_upwind_scalar_face = {
  1,
  "navatala_cfd_primitives_linear_upwind_scalar_face",
  "opencl",
  "navatala_cfd_primitives_linear_upwind_scalar_face",
  "kernel:opencl:navatala_cfd_primitives_linear_upwind_scalar_face",
  "abi-r1:opencl:navatala_cfd_primitives_linear_upwind_scalar_face",
  "abi-r1:opencl:navatala_cfd_primitives_linear_upwind_scalar_face",
  18,
  kAbiArgs_opencl_navatala_cfd_primitives_linear_upwind_scalar_face
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_phi_from_u[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_phi_from_u = {
  1,
  "navatala_cfd_primitives_phi_from_u",
  "opencl",
  "navatala_cfd_primitives_phi_from_u",
  "kernel:opencl:navatala_cfd_primitives_phi_from_u",
  "abi-r1:opencl:navatala_cfd_primitives_phi_from_u",
  "abi-r1:opencl:navatala_cfd_primitives_phi_from_u",
  15,
  kAbiArgs_opencl_navatala_cfd_primitives_phi_from_u
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_phi_from_u_boundary[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_phi_from_u_boundary = {
  1,
  "navatala_cfd_primitives_phi_from_u_boundary",
  "opencl",
  "navatala_cfd_primitives_phi_from_u_boundary",
  "kernel:opencl:navatala_cfd_primitives_phi_from_u_boundary",
  "abi-r1:opencl:navatala_cfd_primitives_phi_from_u_boundary",
  "abi-r1:opencl:navatala_cfd_primitives_phi_from_u_boundary",
  14,
  kAbiArgs_opencl_navatala_cfd_primitives_phi_from_u_boundary
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_phi_from_u_internal[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_phi_from_u_internal = {
  1,
  "navatala_cfd_primitives_phi_from_u_internal",
  "opencl",
  "navatala_cfd_primitives_phi_from_u_internal",
  "kernel:opencl:navatala_cfd_primitives_phi_from_u_internal",
  "abi-r1:opencl:navatala_cfd_primitives_phi_from_u_internal",
  "abi-r1:opencl:navatala_cfd_primitives_phi_from_u_internal",
  11,
  kAbiArgs_opencl_navatala_cfd_primitives_phi_from_u_internal
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_phig_gravity[] = {
  { "ghf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "snGradRho", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhig", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_phig_gravity = {
  1,
  "navatala_cfd_primitives_phig_gravity",
  "opencl",
  "navatala_cfd_primitives_phig_gravity",
  "kernel:opencl:navatala_cfd_primitives_phig_gravity",
  "abi-r1:opencl:navatala_cfd_primitives_phig_gravity",
  "abi-r1:opencl:navatala_cfd_primitives_phig_gravity",
  6,
  kAbiArgs_opencl_navatala_cfd_primitives_phig_gravity
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho",
  "opencl",
  "navatala_cfd_primitives_r_a_u_from_dt_rho",
  "kernel:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho",
  4,
  kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "opencl",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "kernel:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_integrated",
  5,
  kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "opencl",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "kernel:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_sp",
  5,
  kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated = {
  1,
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "opencl",
  "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "kernel:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated",
  6,
  kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx = {
  1,
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "opencl",
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "kernel:opencl:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_u_eqn_approx",
  19,
  kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp = {
  1,
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "opencl",
  "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "kernel:opencl:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  "abi-r1:opencl:navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp",
  20,
  kAbiArgs_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_rho_from_alpha[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRho", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_rho_from_alpha = {
  1,
  "navatala_cfd_primitives_rho_from_alpha",
  "opencl",
  "navatala_cfd_primitives_rho_from_alpha",
  "kernel:opencl:navatala_cfd_primitives_rho_from_alpha",
  "abi-r1:opencl:navatala_cfd_primitives_rho_from_alpha",
  "abi-r1:opencl:navatala_cfd_primitives_rho_from_alpha",
  4,
  kAbiArgs_opencl_navatala_cfd_primitives_rho_from_alpha
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_rho_phi_from_rho_and_phi[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_rho_phi_from_rho_and_phi = {
  1,
  "navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "opencl",
  "navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "kernel:opencl:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "abi-r1:opencl:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  "abi-r1:opencl:navatala_cfd_primitives_rho_phi_from_rho_and_phi",
  9,
  kAbiArgs_opencl_navatala_cfd_primitives_rho_phi_from_rho_and_phi
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_primitives_sn_grad_scalar[] = {
  { "cellVal", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceVal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSnGrad", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_primitives_sn_grad_scalar = {
  1,
  "navatala_cfd_primitives_sn_grad_scalar",
  "opencl",
  "navatala_cfd_primitives_sn_grad_scalar",
  "kernel:opencl:navatala_cfd_primitives_sn_grad_scalar",
  "abi-r1:opencl:navatala_cfd_primitives_sn_grad_scalar",
  "abi-r1:opencl:navatala_cfd_primitives_sn_grad_scalar",
  8,
  kAbiArgs_opencl_navatala_cfd_primitives_sn_grad_scalar
};

bool tryGetKernelAbiManifest_opencl_cfd_primitives(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_average_face_scalar_to_cell") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_average_face_scalar_to_cell;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_evaluate_scalar_bc") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_evaluate_scalar_bc;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_evaluate_vector_bc") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_evaluate_vector_bc;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_face_flux_csr") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_div_face_flux_csr;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_face_flux_csr_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_div_face_flux_csr_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_face_flux_vector_csr") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_div_face_flux_vector_csr;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_face_scalar_product") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_face_scalar_product;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_grad_vol_scalar_gauss") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_grad_vol_scalar_gauss;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_grad_vol_vector_gauss") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_grad_vol_vector_gauss;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_interp_scalar_face") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_interp_scalar_face;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_interp_scalar_face_all") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_interp_scalar_face_all;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_interp_vector_face") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_interp_vector_face;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_linear_upwind_scalar_face") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_linear_upwind_scalar_face;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phi_from_u") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_phi_from_u;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phi_from_u_boundary") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_phi_from_u_boundary;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phi_from_u_internal") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_phi_from_u_internal;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phig_gravity") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_phig_gravity;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_rho_from_alpha") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_rho_from_alpha;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_rho_phi_from_rho_and_phi") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_rho_phi_from_rho_and_phi;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_sn_grad_scalar") {
    out = &kAbiManifest_opencl_navatala_cfd_primitives_sn_grad_scalar;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_cfd_primitives(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_average_face_scalar_to_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_average_face_scalar_to_cell";
    std::string_view sv(k_opencl_navatala_cfd_primitives_average_face_scalar_to_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_evaluate_scalar_bc") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_evaluate_scalar_bc";
    std::string_view sv(k_opencl_navatala_cfd_primitives_evaluate_scalar_bc);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_evaluate_vector_bc") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_evaluate_vector_bc";
    std::string_view sv(k_opencl_navatala_cfd_primitives_evaluate_vector_bc);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_face_flux_csr") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_csr";
    std::string_view sv(k_opencl_navatala_cfd_primitives_div_face_flux_csr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_face_flux_csr_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_csr_f64";
    std::string_view sv(k_opencl_navatala_cfd_primitives_div_face_flux_csr_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_face_flux_vector_csr") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_div_face_flux_vector_csr";
    std::string_view sv(k_opencl_navatala_cfd_primitives_div_face_flux_vector_csr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2";
    std::string_view sv(k_opencl_navatala_cfd_primitives_div_mu_dev2_t_grad_u_from_grad_v2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_face_scalar_product") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_face_scalar_product";
    std::string_view sv(k_opencl_navatala_cfd_primitives_face_scalar_product);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_grad_vol_scalar_gauss") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_grad_vol_scalar_gauss";
    std::string_view sv(k_opencl_navatala_cfd_primitives_grad_vol_scalar_gauss);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_grad_vol_vector_gauss") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_grad_vol_vector_gauss";
    std::string_view sv(k_opencl_navatala_cfd_primitives_grad_vol_vector_gauss);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_interp_scalar_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_interp_scalar_face";
    std::string_view sv(k_opencl_navatala_cfd_primitives_interp_scalar_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_interp_scalar_face_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_interp_scalar_face_all";
    std::string_view sv(k_opencl_navatala_cfd_primitives_interp_scalar_face_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_interp_vector_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_interp_vector_face";
    std::string_view sv(k_opencl_navatala_cfd_primitives_interp_vector_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_linear_upwind_scalar_face") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_linear_upwind_scalar_face";
    std::string_view sv(k_opencl_navatala_cfd_primitives_linear_upwind_scalar_face);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phi_from_u") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u";
    std::string_view sv(k_opencl_navatala_cfd_primitives_phi_from_u);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phi_from_u_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u_boundary";
    std::string_view sv(k_opencl_navatala_cfd_primitives_phi_from_u_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phi_from_u_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_phi_from_u_internal";
    std::string_view sv(k_opencl_navatala_cfd_primitives_phi_from_u_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_phig_gravity") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_phig_gravity";
    std::string_view sv(k_opencl_navatala_cfd_primitives_phig_gravity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho";
    std::string_view sv(k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_integrated";
    std::string_view sv(k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_integrated);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_sp";
    std::string_view sv(k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated";
    std::string_view sv(k_opencl_navatala_cfd_primitives_r_a_u_from_dt_rho_sp_integrated);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_u_eqn_approx";
    std::string_view sv(k_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp";
    std::string_view sv(k_opencl_navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_rho_from_alpha") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_rho_from_alpha";
    std::string_view sv(k_opencl_navatala_cfd_primitives_rho_from_alpha);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_rho_phi_from_rho_and_phi") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_rho_phi_from_rho_and_phi";
    std::string_view sv(k_opencl_navatala_cfd_primitives_rho_phi_from_rho_and_phi);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_primitives_sn_grad_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_primitives_sn_grad_scalar";
    std::string_view sv(k_opencl_navatala_cfd_primitives_sn_grad_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

