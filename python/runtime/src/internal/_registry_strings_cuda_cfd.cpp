// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_cfd` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_navatala_cfd_boundary_force_partials = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_boundary_force_partials(const float* pAllFaces, const float* sfComponent, const int* counts, float* outPartials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  __shared__ float tmp[256];
  float v = __uint_as_float(0x00000000u);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[0]) {
    int faceIdx = (counts[1] + ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))));
    float pVal = pAllFaces[faceIdx];
    float sfVal = sfComponent[faceIdx];
    v = (pVal * sfVal);
  }
  tmp[((int)((int)(threadIdx.x)))] = v;
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 128) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 128)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 64) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 64)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 32) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 32)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 16) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 16)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 8) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 8)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 4) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 4)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 2) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 2)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 1) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 1)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) == 0) {
    outPartials[((int)((int)(blockIdx.x)))] = tmp[0];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pack_owner_values = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pack_owner_values(const float* x, const int* owner, const int* procFaces, const int* counts, float* outSend) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    int face = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int o = owner[face];
    float v = x[o];
    outSend[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = v;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scatter_values_and_mask = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scatter_values_and_mask(const int* procFaces, const float* recv, const int* counts, float* bcVal, int* bcMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    int face = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    bcVal[face] = recv[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    bcMask[face] = 1;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pack_mu_grad_owner = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pack_mu_grad_owner(const float* muCell, const float* gXX, const float* gXY, const float* gXZ, const float* gYX, const float* gYY, const float* gYZ, const float* gZX, const float* gZY, const float* gZZ, const int* owner, const int* procFaces, const int* counts, float* outSend) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int o = owner[f];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 10);
    outSend[base] = muCell[o];
    int b1 = (base + 1);
    outSend[b1] = gXX[o];
    int b2 = (base + 2);
    outSend[b2] = gXY[o];
    int b3 = (base + 3);
    outSend[b3] = gXZ[o];
    int b4 = (base + 4);
    outSend[b4] = gYX[o];
    int b5 = (base + 5);
    outSend[b5] = gYY[o];
    int b6 = (base + 6);
    outSend[b6] = gYZ[o];
    int b7 = (base + 7);
    outSend[b7] = gZX[o];
    int b8 = (base + 8);
    outSend[b8] = gZY[o];
    int b9 = (base + 9);
    outSend[b9] = gZZ[o];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pack_owner_rho_vec3 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pack_owner_rho_vec3(const float* rhoCell, const float* ux, const float* uy, const float* uz, const int* owner, const int* procFaces, const int* counts, float* outSend) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int o = owner[f];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 4);
    outSend[base] = rhoCell[o];
    int b1 = (base + 1);
    outSend[b1] = ux[o];
    int b2 = (base + 2);
    outSend[b2] = uy[o];
    int b3 = (base + 3);
    outSend[b3] = uz[o];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pack_owner_vec3 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pack_owner_vec3(const float* x, const float* y, const float* z, const int* owner, const int* procFaces, const int* counts, float* outSend) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int o = owner[f];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 3);
    outSend[base] = x[o];
    int b1 = (base + 1);
    outSend[b1] = y[o];
    int b2 = (base + 2);
    outSend[b2] = z[o];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pack_owner_vec3_mu = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pack_owner_vec3_mu(const float* ux, const float* uy, const float* uz, const float* muCell, const int* owner, const int* procFaces, const int* counts, float* outSend) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int o = owner[f];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 4);
    outSend[base] = ux[o];
    int b1 = (base + 1);
    outSend[b1] = uy[o];
    int b2 = (base + 2);
    outSend[b2] = uz[o];
    int b3 = (base + 3);
    outSend[b3] = muCell[o];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scatter_mu_grad_ghost = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scatter_mu_grad_ghost(const float* recvVals, const int* counts, float* muCell, float* gXX, float* gXY, float* gXZ, float* gYX, float* gYY, float* gYZ, float* gZX, float* gZY, float* gZZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int idx = (((int)(counts[1])) + (int)(blockIdx.x * blockDim.x + threadIdx.x));
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 10);
    muCell[idx] = recvVals[base];
    int b1 = (base + 1);
    gXX[idx] = recvVals[b1];
    int b2 = (base + 2);
    gXY[idx] = recvVals[b2];
    int b3 = (base + 3);
    gXZ[idx] = recvVals[b3];
    int b4 = (base + 4);
    gYX[idx] = recvVals[b4];
    int b5 = (base + 5);
    gYY[idx] = recvVals[b5];
    int b6 = (base + 6);
    gYZ[idx] = recvVals[b6];
    int b7 = (base + 7);
    gZX[idx] = recvVals[b7];
    int b8 = (base + 8);
    gZY[idx] = recvVals[b8];
    int b9 = (base + 9);
    gZZ[idx] = recvVals[b9];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scatter_rho_vec3_and_mask = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scatter_rho_vec3_and_mask(const int* procFaces, const float* recvVals, const int* counts, float* rhoBcVal, float* uBcX, float* uBcY, float* uBcZ, unsigned int* rhoBcMask, unsigned int* uBcMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 4);
    rhoBcVal[f] = recvVals[base];
    int b1 = (base + 1);
    uBcX[f] = recvVals[b1];
    int b2 = (base + 2);
    uBcY[f] = recvVals[b2];
    int b3 = (base + 3);
    uBcZ[f] = recvVals[b3];
    rhoBcMask[f] = ((unsigned int)(1));
    uBcMask[f] = ((unsigned int)(1));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scatter_vec3_and_mask = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scatter_vec3_and_mask(const int* procFaces, const float* recvVals, const int* counts, float* bcX, float* bcY, float* bcZ, unsigned int* bcMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 3);
    int b1 = (base + 1);
    int b2 = (base + 2);
    bcX[f] = recvVals[base];
    bcY[f] = recvVals[b1];
    bcZ[f] = recvVals[b2];
    bcMask[f] = ((unsigned int)(1));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scatter_vec3_mu_and_mask = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scatter_vec3_mu_and_mask(const int* procFaces, const float* recvVals, const int* counts, float* bcX, float* bcY, float* bcZ, float* muBc, unsigned int* bcMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int base = ((int)(blockIdx.x * blockDim.x + threadIdx.x) * 4);
    bcX[f] = recvVals[base];
    int b1 = (base + 1);
    bcY[f] = recvVals[b1];
    int b2 = (base + 2);
    bcZ[f] = recvVals[b2];
    int b3 = (base + 3);
    muBc[f] = recvVals[b3];
    bcMask[f] = ((unsigned int)(1));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_add_vol_vector = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_add_vol_vector(const float* ax, const float* ay, const float* az, const float* bx, const float* by_, const float* bz, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + bx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ay[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + by_[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (az[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + bz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_div_rho_phi_u_explicit = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_div_rho_phi_u_explicit(const float* phi, const float* rhoCell, const float* ux, const float* uy, const float* uz, const int* owner, const int* neighbour, const float* weights, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const float* rhoBcVal, const unsigned int* rhoBcMask, const float* uBcX, const float* uBcY, const float* uBcZ, const unsigned int* uBcMask, const int* params, float* outX, float* outY, float* outZ) {
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
        float flux = phi[f];
        int o = owner[f];
        float rhoF = rhoCell[o];
        float ufx = ux[o];
        float ufy = uy[o];
        float ufz = uz[o];
        if (f < params[2]) {
          int n = neighbour[f];
          float w0 = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - w0);
          rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
          ufx = ((w0 * ux[o]) + (iw * ux[n]));
          ufy = ((w0 * uy[o]) + (iw * uy[n]));
          ufz = ((w0 * uz[o]) + (iw * uz[n]));
        } else {
          unsigned int rm = rhoBcMask[f];
          if (rm == 1u) {
            float w0 = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - w0);
            rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[f]));
          }
          if (rm == 2u) {
            rhoF = rhoBcVal[f];
          }
          unsigned int um = uBcMask[f];
          if (um == 1u) {
            float w0 = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - w0);
            ufx = ((w0 * ux[o]) + (iw * uBcX[f]));
            ufy = ((w0 * uy[o]) + (iw * uBcY[f]));
            ufz = ((w0 * uz[o]) + (iw * uBcZ[f]));
          }
          if (um == 2u) {
            ufx = uBcX[f];
            ufy = uBcY[f];
            ufz = uBcZ[f];
          }
        }
        float rhoPhi = (rhoF * flux);
        sumX = (sumX + (s * (rhoPhi * ufx)));
        sumY = (sumY + (s * (rhoPhi * ufy)));
        sumZ = (sumZ + (s * (rhoPhi * ufz)));
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
const char* k_cuda_navatala_cfd_hby_a_from_u_and_terms = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_hby_a_from_u_and_terms(const float* ux, const float* uy, const float* uz, const float* lapx, const float* lapy, const float* lapz, const float* divx, const float* divy, const float* divz, const float* extrax, const float* extray, const float* extraz, const float* rau, const int* counts, const int* modeBuf, float* outx, float* outy, float* outz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float tx = lapx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ty = lapy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float tz = lapz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if ((modeBuf[0] == 1) || (modeBuf[0] == 3)) {
      tx = (tx - divx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      ty = (ty - divy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      tz = (tz - divz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    }
    if ((modeBuf[0] == 2) || (modeBuf[0] == 3)) {
      tx = (tx + extrax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      ty = (ty + extray[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      tz = (tz + extraz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    }
    float a = rau[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (a * tx));
    outy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (a * ty));
    outz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (a * tz));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_apply_sp_to_hby_a = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_apply_sp_to_hby_a(float* hx, float* hy, float* hz, const float* ux, const float* uy, const float* uz, const float* rAU, const float* sp, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float fac = (sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    hx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (fac * ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    hy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (fac * uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    hz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (fac * uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_laplacian_scalar_explicit = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_laplacian_scalar_explicit(const float* muCell, const float* phiCell, const int* owner, const int* neighbour, const float* weights, const float* magSf, const float* deltaCoeffs, const int* offsets, const int* faceIdx, const int* sign, const float* vol, const float* bcMu, const unsigned int* bcMuMask, const float* bcPhi, const unsigned int* bcPhiMask, const int* params, float* outL) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float phiC = phiCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float muC = muCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float sum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        int sgn = sign[k];
        int o = owner[f];
        float phiOther = phiCell[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            phiOther = phiCell[n];
            float w0 = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            if (bcPhiMask[f] != 0u) {
              phiOther = bcPhi[f];
            }
            if (bcMuMask[f] != 0u) {
              muF = bcMu[f];
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sum = (sum + (coeff * (phiOther - phiC)));
      }
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outL[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sum * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_laplacian_vector_explicit = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_laplacian_vector_explicit(const float* muCell, const float* muBc, const float* ux, const float* uy, const float* uz, const int* owner, const int* neighbour, const float* weights, const float* magSf, const float* deltaCoeffs, const float* corrX, const float* corrY, const float* corrZ, const float* gradRow0, const float* gradRow1, const float* gradRow2, const int* offsets, const int* faceIdx, const int* sign, const float* vol, const float* bcX, const float* bcY, const float* bcZ, const int* bcMask, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float ucx = ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ucy = uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ucz = uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float muC = muCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
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
        int sgn = sign[k];
        int o = owner[f];
        float otherX = ux[o];
        float otherY = uy[o];
        float otherZ = uz[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            otherX = ux[n];
            otherY = uy[n];
            otherZ = uz[n];
            float w0 = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            muF = muBc[f];
            if (bcMask[f] == 1) {
              float w0 = weights[f];
              float iw = (__uint_as_float(0x3f800000u) - w0);
              muF = ((w0 * muC) + (iw * muBc[f]));
              otherX = bcX[f];
              otherY = bcY[f];
              otherZ = bcZ[f];
            } else {
              if (bcMask[f] != 0) {
                otherX = bcX[f];
                otherY = bcY[f];
                otherZ = bcZ[f];
              }
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sumX = (sumX + (coeff * (otherX - ucx)));
        sumY = (sumY + (coeff * (otherY - ucy)));
        sumZ = (sumZ + (coeff * (otherZ - ucz)));
        float kx = corrX[f];
        float ky = corrY[f];
        float kz = corrZ[f];
        float gxxF = gradRow0[((o)*3 + 0)];
        float gxyF = gradRow0[((o)*3 + 1)];
        float gxzF = gradRow0[((o)*3 + 2)];
        float gyxF = gradRow1[((o)*3 + 0)];
        float gyyF = gradRow1[((o)*3 + 1)];
        float gyzF = gradRow1[((o)*3 + 2)];
        float gzxF = gradRow2[((o)*3 + 0)];
        float gzyF = gradRow2[((o)*3 + 1)];
        float gzzF = gradRow2[((o)*3 + 2)];
        if (f < params[2]) {
          int n2 = neighbour[f];
          float w1 = weights[f];
          float iw1 = (__uint_as_float(0x3f800000u) - w1);
          gxxF = ((w1 * gradRow0[((o)*3 + 0)]) + (iw1 * gradRow0[((n2)*3 + 0)]));
          gxyF = ((w1 * gradRow0[((o)*3 + 1)]) + (iw1 * gradRow0[((n2)*3 + 1)]));
          gxzF = ((w1 * gradRow0[((o)*3 + 2)]) + (iw1 * gradRow0[((n2)*3 + 2)]));
          gyxF = ((w1 * gradRow1[((o)*3 + 0)]) + (iw1 * gradRow1[((n2)*3 + 0)]));
          gyyF = ((w1 * gradRow1[((o)*3 + 1)]) + (iw1 * gradRow1[((n2)*3 + 1)]));
          gyzF = ((w1 * gradRow1[((o)*3 + 2)]) + (iw1 * gradRow1[((n2)*3 + 2)]));
          gzxF = ((w1 * gradRow2[((o)*3 + 0)]) + (iw1 * gradRow2[((n2)*3 + 0)]));
          gzyF = ((w1 * gradRow2[((o)*3 + 1)]) + (iw1 * gradRow2[((n2)*3 + 1)]));
          gzzF = ((w1 * gradRow2[((o)*3 + 2)]) + (iw1 * gradRow2[((n2)*3 + 2)]));
        }
        float corrUx = (((kx * gxxF) + (ky * gxyF)) + (kz * gxzF));
        float corrUy = (((kx * gyxF) + (ky * gyyF)) + (kz * gyzF));
        float corrUz = (((kx * gzxF) + (ky * gzyF)) + (kz * gzzF));
        float fluxScale = (muF * magSf[f]);
        float sgnF = ((float)(sgn));
        sumX = (sumX + ((sgnF * fluxScale) * corrUx));
        sumY = (sumY + ((sgnF * fluxScale) * corrUy));
        sumZ = (sumZ + ((sgnF * fluxScale) * corrUz));
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
const char* k_cuda_navatala_cfd_predictor_hby_a_from_source = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_predictor_hby_a_from_source(const float* rAU, const float* sourceX, const float* sourceY, const float* sourceZ, const int* counts1, float* outHbyAX, float* outHbyAY, float* outHbyAZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts1[0]) {
    return;
  } else {
    float r = rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outHbyAX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * sourceX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outHbyAY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * sourceY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outHbyAZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * sourceZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_predictor_inverse_diag = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_predictor_inverse_diag(const float* inputDiag, const int* counts1, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts1[0]) {
    return;
  } else {
    float a = inputDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (a != __uint_as_float(0x00000000u)) {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (__uint_as_float(0x3f800000u) / a);
    } else {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_predictor_phi_hby_a_fuse = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_predictor_phi_hby_a_fuse(const float* fluxHbyA, const float* rhoRauFace, const float* ddtCorr, const int* counts1, float* outPhiHbyA) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts1[0]) {
    return;
  } else {
    float f = fluxHbyA[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float r = rhoRauFace[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float d = ddtCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outPhiHbyA[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (f + (r * d));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_axpy_cells = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_axpy_cells(const float* x, const int* counts, const float* paramsF, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float xVal = x[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float yVal = y[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float result = (yVal + (paramsF[0] * xVal));
    y[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = result;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_ddt_diagonal = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_ddt_diagonal(const float* rho, const float* vol, const int* counts, const float* paramsF, float* outDiag) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float rhoVal = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float volVal = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float diag = ((rhoVal * volVal) * paramsF[0]);
    outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = diag;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_div_upwind_coeffs = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_div_upwind_coeffs(const float* weights, const float* flux, const int* counts, float* outLower, float* outUpper) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float phi = flux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float low = ((__uint_as_float(0x00000000u) - w) * phi);
    float up = (low + phi);
    outLower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = low;
    outUpper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = up;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_fvm_sp = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_fvm_sp(const float* sp, const float* vol, const int* counts, float* outDiag) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float volVal = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float contrib = (spVal * volVal);
    float prev = outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float newDiag = (prev + contrib);
    outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newDiag;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_fvm_su_sp = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_fvm_su_sp(const float* sp, const float* psi, const float* vol, const int* counts, float* outDiag, float* outSource) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float psiVal = psi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float volVal = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float spVol = (spVal * volVal);
    if (spVal >= __uint_as_float(0x00000000u)) {
      float prevDiag = outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float newDiag = (prevDiag + spVol);
      outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newDiag;
    } else {
      float prevSrc = outSource[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float contrib = (spVol * psiVal);
      float newSrc = (prevSrc - contrib);
      outSource[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newSrc;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_laplacian_coeffs = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_laplacian_coeffs(const float* gamma, const int* owner, const int* neighbour, const float* weights, const float* deltaCoeffs, const float* magSf, const int* counts, float* outUpper, float* outLower) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float gammaF = gamma[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float coeff = (deltaCoeffs[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (gammaF * magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    outUpper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = coeff;
    outLower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = coeff;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_laplacian_coeffs_accum = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_laplacian_coeffs_accum(const float* gamma, const float* deltaCoeffs, const float* magSf, const int* counts, float* outUpper, float* outLower) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float coeff = (deltaCoeffs[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (gamma[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    float prevUpper = outUpper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float prevLower = outLower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outUpper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (prevUpper + coeff);
    outLower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (prevLower + coeff);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_linear_upwind_correction_source = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_linear_upwind_correction_source(const float* luFaceVals, const float* flux, const int* owner, const int* neighbour, const float* psi, const int* offsets, const int* faceIdx, const float* signF, const int* counts, float* outSource) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float sum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        float fluxVal = flux[f];
        float luVal = luFaceVals[f];
        float sgnF = signF[k];
        float correction = (fluxVal * luVal);
        sum = (sum + (sgnF * correction));
      }
    }
    float prev = outSource[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outSource[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (prev - sum);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_matrix_relax = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_matrix_relax(const float* diagOld, const float* sumOff, const float* psi, const int* counts, const float* paramsF, float* diag, float* source) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float d0 = diagOld[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float sOff = sumOff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float x = psi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float negD0 = (__uint_as_float(0x00000000u) - d0);
    float absD0 = (((d0 >= __uint_as_float(0x00000000u))) ? (d0) : (negD0));
    float maxVal = (((absD0 >= sOff)) ? (absD0) : (sOff));
    float newDiag = (maxVal / paramsF[0]);
    float srcPrev = source[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float delta = (newDiag - d0);
    float srcNew = (srcPrev + (delta * x));
    diag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = newDiag;
    source[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = srcNew;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_neg_sum_diag = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_neg_sum_diag(const float* upper, const float* lower, const int* offsets, const int* faceIdx, const int* sign, const int* counts, float* outDiag) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float sum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int sgn = sign[k];
        if (sgn > 0) {
          sum = (sum - upper[f]);
        } else {
          sum = (sum - lower[f]);
        }
      }
    }
    float prev = outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outDiag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (prev + sum);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pack_boundary_adjacent_cells = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pack_boundary_adjacent_cells(const float* cellField, const int* indexMap, const int* counts, float* outPacked) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int cellIdx = indexMap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float val = cellField[cellIdx];
    outPacked[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = val;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scalar_jacobi_clamp_min = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scalar_jacobi_clamp_min(const float* lowerBound, const unsigned int* counts, float* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float lb = lowerBound[0];
    float xi = x[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (xi > lb) {
      x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = xi;
    } else {
      x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = lb;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scalar_jacobi_update = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scalar_jacobi_update(const float* ax, const float* diag, const float* rhs, const float* params, const unsigned int* counts, float* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float absD = abs(d);
    if (absD < smallDiag) {
      if (d >= __uint_as_float(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float xi = x[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float correction = (omega * ((rhs[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)]) / d));
    x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (xi + correction);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scalar_ldu_mat_vec = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scalar_ldu_mat_vec(const float* diag, const float* upper, const float* lower, const float* x, const int* owner, const int* neighbour, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const unsigned int* counts, float* ax) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * x[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
    int beg = ((int)(offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int f = faceIdx[k];
      int sg = sign[k];
      if (((int)(f)) < ((int)(counts[1]))) {
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = s;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scalar_ldu_multi_color_gs_sweep = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scalar_ldu_multi_color_gs_sweep(const float* diag, const float* upper, const float* lower, const float* rhs, const int* owner, const int* neighbour, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const unsigned int* colorOrder, const unsigned int* colorRange, const float* params, const unsigned int* counts, float* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int colorSize = (((int)(colorRange[1])) - ((int)(colorRange[0])));
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= colorSize) {
    return;
  } else {
    int cellIdx = (((int)(colorRange[0])) + (int)(blockIdx.x * blockDim.x + threadIdx.x));
    unsigned int celli = colorOrder[cellIdx];
    int ci = ((int)(celli));
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[ci];
    float absD = abs(d);
    if (absD < smallDiag) {
      if (d >= __uint_as_float(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float s = (d * x[ci]);
    int beg = ((int)(offsets[ci]));
    int c1 = (ci + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int f = faceIdx[k];
      int sg = sign[k];
      if (((int)(f)) < ((int)(counts[1]))) {
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    x[ci] = (x[ci] + (omega * ((rhs[ci] - s) / d)));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_sum_mag_off_diag = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_sum_mag_off_diag(const float* upper, const float* lower, const int* offsets, const int* faceIdx, const int* sign, const int* counts, float* outSumOff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float sum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int sgn = sign[k];
        if (sgn > 0) {
          float val = upper[f];
          float negVal = (__uint_as_float(0x00000000u) - val);
          float absVal = (((val >= __uint_as_float(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        } else {
          float val = lower[f];
          float negVal = (__uint_as_float(0x00000000u) - val);
          float absVal = (((val >= __uint_as_float(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        }
      }
    }
    outSumOff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sum;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_axpy_in_place = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_axpy_in_place(const float* x, const float* alpha, const unsigned int* nRows, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(nRows[0]))) {
    return;
  } else {
    float a = alpha[0];
    float xi = x[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float yi = y[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    y[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (yi + (a * xi));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_compute_scale_factor = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_compute_scale_factor(const float* num, const float* den, float* sf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    float n = num[0];
    float d = den[0];
    float dAbs = abs(d);
    if (dAbs > __uint_as_float(0x0da24260u)) {
      sf[0] = (n / d);
    } else {
      sf[0] = __uint_as_float(0x3f800000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_dot_finalize = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_dot_finalize(const float* partials, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    float s = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(((int)(count[0]))); ++i) {
      s = (s + partials[i]);
    }
    result[0] = s;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_negate_scalar = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_negate_scalar(const float* _input, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    float v = _input[0];
    _output[0] = (-v);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_scale_correction = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_scale_correction(const float* z, const float* r, const float* Az, const float* diagInv, const float* sfPtr, const unsigned int* nRows, float* corr) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(nRows[0]))) {
    return;
  } else {
    float sf = sfPtr[0];
    float zi = z[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float ri = r[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float azi = Az[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float di = diagInv[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float sfz = (sf * zi);
    float sfaz = (sf * azi);
    float residual = (ri - sfaz);
    float corrVal = (sfz + (residual * di));
    corr[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = corrVal;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_dic_build = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_dic_build(const float* diag, const int* edgeU, const int* edgeV, const float* edgeCf, const int* counts, double* rD) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    int nCells = counts[1];
    int nEdges = counts[2];
    for (int c = 0; c < (int)(nCells); ++c) {
      rD[c] = ((double)(diag[c]));
    }
    for (int e = 0; e < (int)(nEdges); ++e) {
      int u = edgeU[e];
      int v = edgeV[e];
      double dl = rD[u];
      if (dl > __longlong_as_double(0x01a56e1fc2f8f359ull)) {
        double cf = ((double)(edgeCf[e]));
        rD[v] = (rD[v] - ((cf * cf) / dl));
      }
    }
    for (int c2 = 0; c2 < (int)(nCells); ++c2) {
      double d = rD[c2];
      if (d > __longlong_as_double(0x01a56e1fc2f8f359ull)) {
        rD[c2] = (__longlong_as_double(0x3ff0000000000000ull) / d);
      } else {
        rD[c2] = __longlong_as_double(0x0000000000000000ull);
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_dic_smooth = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_dic_smooth(const float* b, const float* ax, const double* rD, const int* edgeU, const int* edgeV, const float* edgeCf, const int* counts, float* x, float* z) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    int nCells = counts[1];
    int nEdges = counts[2];
    for (int c = 0; c < (int)(nCells); ++c) {
      z[c] = ((float)((rD[c] * (((double)(b[c])) - ((double)(ax[c]))))));
    }
    for (int e = 0; e < (int)(nEdges); ++e) {
      int v = edgeV[e];
      int u = edgeU[e];
      z[v] = ((float)((((double)(z[v])) - (rD[v] * (((double)(edgeCf[e])) * ((double)(z[u])))))));
    }
    for (int t = 0; t < (int)(nEdges); ++t) {
      int e = ((nEdges - 1) - t);
      int u = edgeU[e];
      int v = edgeV[e];
      z[u] = ((float)((((double)(z[u])) - (rD[u] * (((double)(edgeCf[e])) * ((double)(z[v])))))));
    }
    for (int c2 = 0; c2 < (int)(nCells); ++c2) {
      x[c2] = (x[c2] + z[c2]);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_coarse_jacobi = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_coarse_jacobi(float* x, const float* r, const float* diag, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    float d = diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (d == __uint_as_float(0x00000000u)) {
      return;
    } else {
      float omega = __uint_as_float(0x3f4ccccdu);
      x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + (omega * (r[(int)(blockIdx.x * blockDim.x + threadIdx.x)] / d)));
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_coarse_matvec = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_coarse_matvec(const int* edgeU, const int* edgeV, const float* edgeCf, const float* diag, const float* x, float* outAx, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = (((int)(mgCounts[1])) + ((int)(mgCounts[2])));
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= total) {
    return;
  } else {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) < ((int)(mgCounts[1]))) {
      atomicAdd(&outAx[(int)(blockIdx.x * blockDim.x + threadIdx.x)], (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * x[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
    } else {
      int e = ((int)(blockIdx.x * blockDim.x + threadIdx.x) - ((int)(mgCounts[1])));
      int u = edgeU[e];
      int v = edgeV[e];
      float cf = edgeCf[e];
      atomicAdd(&outAx[u], (cf * x[v]));
      atomicAdd(&outAx[v], (cf * x[u]));
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_coarse_rhs = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_coarse_rhs(const float* b, const float* ax, float* r, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    r[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (b[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_find_best_match = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_find_best_match(const unsigned int* csrOffsets, const unsigned int* csrFaceIdx, const int* csrSign, const float* strength, const int* owner, const int* nei, int* bestNeighbor, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float bestS = __uint_as_float(0xbf800000u);
    int bestN = -1;
    unsigned int beg = csrOffsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    unsigned int end = csrOffsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      unsigned int f = csrFaceIdx[k];
      float s = strength[((int)(f))];
      if (s > bestS) {
        bestS = s;
        int sg = csrSign[k];
        if (sg > 0) {
          bestN = nei[((int)(f))];
        } else {
          bestN = owner[((int)(f))];
        }
      }
    }
    bestNeighbor[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = bestN;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_pair_match = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_pair_match(const int* bestNeighbor, int* aggMap, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int n = bestNeighbor[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (n >= 0) {
      if ((int)(blockIdx.x * blockDim.x + threadIdx.x) < n) {
        aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (int)(blockIdx.x * blockDim.x + threadIdx.x);
      } else {
        aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = n;
      }
    } else {
      aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_prolong_add = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_prolong_add(const float* xCoarse, const int* prolongRowOffsets, const int* prolongCols, const float* prolongVals, float* zFine, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    float accum = __uint_as_float(0x00000000u);
    int beg = prolongRowOffsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = prolongRowOffsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      accum = (accum + (prolongVals[k] * xCoarse[col]));
    }
    zFine[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (zFine[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + accum);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_renumber_aggregates = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_renumber_aggregates(int* aggMap, int* counter, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    if (aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] == (int)(blockIdx.x * blockDim.x + threadIdx.x)) {
      int oldCount = atomicAdd(&(counter[0]), 1);
      aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (-(oldCount + 1));
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_renumber_propagate = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_renumber_propagate(int* aggMap, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int root = aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (root < 0) {
      aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (-(root + 1));
    } else {
      int rootVal = aggMap[root];
      if (rootVal < 0) {
        aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (-(rootVal + 1));
      } else {
        aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = rootVal;
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_resolve_chains = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_resolve_chains(int* aggMap, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int root = aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (root >= 0) {
      if (root < ((int)(counts[0]))) {
        if (root != (int)(blockIdx.x * blockDim.x + threadIdx.x)) {
          int next = aggMap[root];
          if (next >= 0) {
            if (next < ((int)(counts[0]))) {
              aggMap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = next;
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_restrict_sum = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_restrict_sum(const float* rFine, const int* prolongRowOffsets, const int* prolongCols, const float* prolongVals, float* rCoarse, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    int beg = prolongRowOffsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = prolongRowOffsets[c1];
    float v = rFine[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      atomicAdd(&rCoarse[col], (prolongVals[k] * v));
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_strength_of_connection = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_strength_of_connection(const float* diag, const float* upper, const int* owner, const int* nei, float* strength, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[1]))) {
    return;
  } else {
    int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int n = nei[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float dO = abs(diag[o]);
    float dN = abs(diag[n]);
    float denomSq = (dO * dN);
    if (denomSq > __uint_as_float(0x0da24260u)) {
      strength[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (abs(upper[(int)(blockIdx.x * blockDim.x + threadIdx.x)]) / sqrt(denomSq));
    } else {
      strength[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mg_zero_coarse = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mg_zero_coarse(float* x, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_phi_correct = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_phi_correct(const float* phiIn, const float* pCell, const int* owner, const int* neighbour, const float* magSf, const float* deltaCoeffs, const float* coeff, const float* bcVal, const int* bcMask, const float* bcSnGrad, const int* bcSnGradMask, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* phiOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float po = pCell[o];
    float other = po;
    float snGrad = __uint_as_float(0x00000000u);
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      other = pCell[n];
    } else {
      if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0) {
        other = bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      }
    }
    snGrad = (deltaCoeffs[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (other - po));
    if (bcSnGradMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0) {
      snGrad = bcSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    }
    float gradFace = (magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * snGrad);
    phiOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phiIn[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - ((paramsF[0] * (coeff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * gradFace)) + faceFluxCorrection[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_phi_correct_boundary = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_phi_correct_boundary(const float* phiIn, const float* pCell, const int* owner, const float* magSf, const float* deltaCoeffs, const float* coeff, const float* bcVal, const int* bcMask, const float* bcSnGrad, const int* bcSnGradMask, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* phiOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int f = (counts[1] + ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))));
  if (f >= counts[0]) {
    return;
  } else {
    int o = owner[f];
    float po = pCell[o];
    float other = po;
    if (bcMask[f] != 0) {
      other = bcVal[f];
    }
    float snGrad = (deltaCoeffs[f] * (other - po));
    if (bcSnGradMask[f] != 0) {
      snGrad = bcSnGrad[f];
    }
    float gradFace = (magSf[f] * snGrad);
    phiOut[f] = (phiIn[f] - ((paramsF[0] * (coeff[f] * gradFace)) + faceFluxCorrection[f]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_phi_correct_internal = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_phi_correct_internal(const float* phiIn, const float* pCell, const int* owner, const int* neighbour, const float* magSf, const float* deltaCoeffs, const float* coeff, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* phiOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float po = pCell[o];
    float pn = pCell[n];
    float snGrad = (deltaCoeffs[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (pn - po));
    float gradFace = (magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * snGrad);
    phiOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phiIn[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - ((paramsF[0] * (coeff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * gradFace)) + faceFluxCorrection[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_axpy_in_place = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_axpy_in_place(const float* x, float* y, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    float yi = y[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    y[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (yi + (paramsF[0] * x[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_bc_dirichlet_face_flux = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_bc_dirichlet_face_flux(const float* cf, const float* bcVal, const int* bcMask, const int* counts, float* outFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    float flux = __uint_as_float(0x00000000u);
    if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 2) {
      flux = (cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    }
    outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = flux;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_bc_sn_grad_face_flux = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_bc_sn_grad_face_flux(const float* cf, const float* delta, const float* bcSnGrad, const int* bcSnGradMask, const int* counts, float* outFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    float d = delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float sg = bcSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float flux = __uint_as_float(0x00000000u);
    if (bcSnGradMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0) {
      if (d != __uint_as_float(0x00000000u)) {
        flux = (cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (sg / d));
      }
    }
    outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = flux;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_coeff_to_cf_in_place = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_coeff_to_cf_in_place(float* cf, const float* magSf, const float* delta, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    float v = cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((v * magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) * delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_diag_from_cf = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_diag_from_cf(const float* cf, const int* offsets, const int* faceIdx, const float* vol, const int* counts, float* diag) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + cf[f]);
    }
    diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (__uint_as_float(0x00000000u) - s);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_dic_apply = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_dic_apply(const float* r, const float* rD, const float* upper, const int* owner, const int* neighbour, const int* counts, float* z) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    z[c] = (rD[c] * r[c]);
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float zn = z[n];
    z[n] = (zn - ((rD[n] * upper[f]) * z[o]));
  }
  for (int t = 0; t < (int)(counts[2]); ++t) {
    int f2 = ((counts[2] - 1) - t);
    int o2 = owner[f2];
    int n2 = neighbour[f2];
    float zo = z[o2];
    z[o2] = (zo - ((rD[o2] * upper[f2]) * z[n2]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_dic_build_reciprocal_d = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_dic_build_reciprocal_d(const float* diag, const float* upper, const int* owner, const int* neighbour, const int* counts, float* rD) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    rD[c] = diag[c];
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float u = upper[f];
    float rdo = rD[o];
    float rdn = rD[n];
    rD[n] = (rdn - ((u * u) / rdo));
  }
  for (int c2 = 0; c2 < (int)(counts[0]); ++c2) {
    float v = rD[c2];
    if (v != __uint_as_float(0x00000000u)) {
      rD[c2] = (__uint_as_float(0x3f800000u) / v);
    } else {
      rD[c2] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_dot_partials = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_dot_partials(const float* a, const float* b, const int* counts, float* outPartials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double tmp[256];
  float v = __uint_as_float(0x00000000u);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[0]) {
    v = (a[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * b[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
  tmp[((int)((int)(threadIdx.x)))] = v;
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 128) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 128)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 64) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 64)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 32) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 32)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 16) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 16)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 8) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 8)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 4) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 4)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 2) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 2)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 1) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 1)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) == 0) {
    outPartials[((int)((int)(blockIdx.x)))] = tmp[0];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_face_flux = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_face_flux(const float* x, const int* owner, const int* neighbour, const float* cf, const float* bcVal, const int* bcMask, const int* counts, float* outFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float xo = x[o];
    float other = xo;
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      other = x[n];
    } else {
      int m = bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if ((m == 1) || (m == 3)) {
        other = bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      } else {
        if (m == 2) {
          other = __uint_as_float(0x00000000u);
        } else {
          if (m == 4) {
            outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
            return;
          }
        }
      }
    }
    outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (other - xo));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_inv_diag = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_inv_diag(const float* r, const float* diag, const int* counts, float* z) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    float d = diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float v = r[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (d != __uint_as_float(0x00000000u)) {
      z[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (v / d);
    } else {
      z[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = v;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_laplacian_from_face_flux = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_laplacian_from_face_flux(const float* faceFlux, const int* offsets, const int* faceIdx, const float* signF, const float* vol, const int* counts, float* outLap) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + (signF[k] * faceFlux[f]));
    }
    outLap[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = s;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mul_by_vol_in_place = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mul_by_vol_in_place(float* b, const float* vol, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float v = b[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    b[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (v * vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_ref_add_ax = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_ref_add_ax(const float* diag, const float* x, float* outAx, const int* refCell) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int c = refCell[0];
  if (c < 0) {
    return;
  } else {
    float cur = outAx[c];
    outAx[c] = (cur + (diag[c] * x[c]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_ref_add_b = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_ref_add_b(const float* diag, float* b, const int* refCell, const float* refVal) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int c = refCell[0];
  if (c < 0) {
    return;
  } else {
    float cur = b[c];
    b[c] = (cur + (diag[c] * refVal[0]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_rhs_sub = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_rhs_sub(const float* b, const float* ax, const int* counts, float* r) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    r[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (b[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - ax[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_shift_in_place = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_shift_in_place(float* x, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float xi = x[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    x[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (xi + paramsF[0]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_sum_abs_partials = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_sum_abs_partials(const float* a, const int* counts, float* outPartials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double tmp[256];
  float v = __uint_as_float(0x00000000u);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[0]) {
    float ai = a[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (ai >= __uint_as_float(0x00000000u)) {
      v = ai;
    } else {
      v = (__uint_as_float(0x00000000u) - ai);
    }
  }
  tmp[((int)((int)(threadIdx.x)))] = v;
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 128) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 128)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 64) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 64)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 32) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 32)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 16) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 16)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 8) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 8)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 4) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 4)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 2) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 2)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) < 1) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 1)]);
  }
  __syncthreads();
  if (((int)((int)(threadIdx.x))) == 0) {
    outPartials[((int)((int)(blockIdx.x)))] = tmp[0];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_upper_from_cf = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_upper_from_cf(const float* cf, const int* counts, float* upper) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]) {
    return;
  } else {
    upper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
  }
}

)kernel";
const char* k_cuda_navatala_cfd_xpay_in_place = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_xpay_in_place(const float* z, const float* pIn, float* pOut, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= counts[0]) {
    return;
  } else {
    pOut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (z[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + (paramsF[0] * pIn[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_dot_finalize_partials = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_dot_finalize_partials(const double* partials, const unsigned int* count, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    double s = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(((int)(count[0]))); ++i) {
      s = (s + partials[i]);
    }
    result[0] = s;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_negate_scalar = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_negate_scalar(const float* _input, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    float v = _input[0];
    _output[0] = (-v);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pcg_compute_alpha = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pcg_compute_alpha(const double* rz, const double* pAp, float* alpha) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    double rz_val = rz[0];
    double pAp_val = pAp[0];
    if (pAp_val != __longlong_as_double(0x0000000000000000ull)) {
      alpha[0] = ((float)((rz_val / pAp_val)));
    } else {
      alpha[0] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pcg_compute_beta = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pcg_compute_beta(const double* rzNew, const double* rzOld, float* beta) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= 1) {
    return;
  } else {
    double new_val = rzNew[0];
    double old_val = rzOld[0];
    if (old_val != __longlong_as_double(0x0000000000000000ull)) {
      beta[0] = ((float)((new_val / old_val)));
    } else {
      beta[0] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_exact_mat_vec = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_exact_mat_vec(const float* x, const int* owner, const int* neighbour, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const float* upper, const float* lower, const float* diag, const int* counts, float* outAx) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * x[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
    unsigned int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    unsigned int end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      unsigned int f = faceIdx[k];
      if (((int)(f)) < ((int)(counts[2]))) {
        int sg = sign[k];
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    outAx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = s;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_exact_mat_vec_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_exact_mat_vec_f64(const float* x, const int* owner, const int* neighbour, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const double* upper, const double* lower, const double* diag, const int* counts, float* outAx) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    double s = (diag[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ((double)(x[(int)(blockIdx.x * blockDim.x + threadIdx.x)])));
    unsigned int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    unsigned int end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      unsigned int f = faceIdx[k];
      if (((int)(f)) < ((int)(counts[2]))) {
        int sg = sign[k];
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * ((double)(x[nb]))));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * ((double)(x[ow]))));
        }
      }
    }
    outAx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ((float)(s));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_face_flux_boundary = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_face_flux_boundary(const float* x, const int* owner, const float* cf, const float* bcVal, const unsigned int* bcMask, const unsigned int* params, float* faceFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    int gf = (((int)(params[1])) + (int)(blockIdx.x * blockDim.x + threadIdx.x));
    int o = owner[gf];
    float xo = x[o];
    unsigned int m = bcMask[gf];
    int mInt = ((int)(m));
    if (mInt == 4) {
      faceFlux[gf] = bcVal[gf];
    } else {
      float other = xo;
      if (mInt == 1) {
        other = bcVal[gf];
      } else {
        if (mInt == 3) {
          other = bcVal[gf];
        } else {
          if (mInt == 2) {
            other = __uint_as_float(0x00000000u);
          }
        }
      }
      faceFlux[gf] = (cf[gf] * (other - xo));
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_face_flux_internal = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_face_flux_internal(const float* x, const int* owner, const int* neighbour, const float* cf, const unsigned int* params, float* faceFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int n = neighbour[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    faceFlux[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (cf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * (x[n] - x[o]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pressure_face_flux_correction_corrected = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pressure_face_flux_correction_corrected(const int* owner, const int* neighbour, const float* weights, const float* gradX, const float* gradY, const float* gradZ, const float* magSf, const float* dCorrX, const float* dCorrY, const float* dCorrZ, const float* rAUf, const unsigned int* params, float* outFaceFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[1]))) {
      outFaceFlux[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
    } else {
      int own = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      int nei = neighbour[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float w = weights[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float omw = (__uint_as_float(0x3f800000u) - w);
      float gx = ((w * gradX[own]) + (omw * gradX[nei]));
      float gy = ((w * gradY[own]) + (omw * gradY[nei]));
      float gz = ((w * gradZ[own]) + (omw * gradZ[nei]));
      float corr = (((dCorrX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * gx) + (dCorrY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * gy)) + (dCorrZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * gz));
      outFaceFlux[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (rAUf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * (magSf[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * corr));
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_pressure_sn_grad = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_pressure_sn_grad(const float* phiHbyA, const int* ownerAll, const float* sfX, const float* sfY, const float* sfZ, const float* magSf, const float* ux, const float* uy, const float* uz, const float* uBcX, const float* uBcY, const float* uBcZ, const unsigned int* uBcMask, const float* rAUf, const unsigned int* snGradMask, const unsigned int* params, float* outSnGrad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[1]))) {
    return;
  } else {
    unsigned int mask = snGradMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (((int)(mask)) == 0) {
      outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
    } else {
      int o = ownerAll[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      if (o < 0) {
        outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
      } else {
        if (o >= ((int)(params[0]))) {
          outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
        } else {
          float uxF = ux[o];
          float uyF = uy[o];
          float uzF = uz[o];
          unsigned int bcm = uBcMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          if (((int)(bcm)) == 2) {
            uxF = uBcX[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
            uyF = uBcY[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
            uzF = uBcZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          }
          float phiU = (((sfX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * uxF) + (sfY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * uyF)) + (sfZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * uzF));
          float rf = rAUf[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          float ms = magSf[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          float denom = (rf * ms);
          float ph = phiHbyA[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          if (abs(denom) < __uint_as_float(0x1e3ce508u)) {
            outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
          } else {
            outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ((ph - (rf * phiU)) / denom);
          }
        }
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_u_correct = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_u_correct(const float* pCell, const int* owner, const int* neighbour, const float* sfX, const float* sfY, const float* sfZ, const float* bcVal, const int* bcMask, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const float* rAU, const float* ux, const float* uy, const float* uz, const int* counts, const float* paramsF, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float gx = __uint_as_float(0x00000000u);
    float gy = __uint_as_float(0x00000000u);
    float gz = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      int o = owner[f];
      float pf = pCell[o];
      if (f < counts[2]) {
        int n = neighbour[f];
        pf = (__uint_as_float(0x3f000000u) * (pCell[o] + pCell[n]));
      } else {
        if (bcMask[f] != 0) {
          pf = bcVal[f];
        }
      }
      float s = sign[k];
      gx = (gx + (s * (sfX[f] * pf)));
      gy = (gy + (s * (sfY[f] * pf)));
      gz = (gz + (s * (sfZ[f] * pf)));
    }
    float invV = (__uint_as_float(0x3f800000u) / vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    float sC = (paramsF[0] * rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    float dUx = ((gx * invV) * sC);
    float dUy = ((gy * invV) * sC);
    float dUz = ((gz * invV) * sC);
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - dUx);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - dUy);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - dUz);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_u_correct_reconstruct = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_u_correct_reconstruct(const float* pCell, const float* rAU, const float* hbx, const float* hby, const float* hbz, const float* rAUf, const float* phig, const int* owner, const int* neighbour, const float* sfX, const float* sfY, const float* sfZ, const float* magSf, const float* deltaCoeffs, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const float* bcVal, const int* bcMask, const float* bcSnGrad, const int* bcSnGradMask, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float a00 = __uint_as_float(0x00000000u);
    float a01 = __uint_as_float(0x00000000u);
    float a02 = __uint_as_float(0x00000000u);
    float a10 = __uint_as_float(0x00000000u);
    float a11 = __uint_as_float(0x00000000u);
    float a12 = __uint_as_float(0x00000000u);
    float a20 = __uint_as_float(0x00000000u);
    float a21 = __uint_as_float(0x00000000u);
    float a22 = __uint_as_float(0x00000000u);
    float bx = __uint_as_float(0x00000000u);
    float by = __uint_as_float(0x00000000u);
    float bz = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int o = owner[f];
        float po = pCell[o];
        float other = po;
        if (f < counts[2]) {
          int n = neighbour[f];
          other = pCell[n];
        } else {
          if (bcMask[f] != 0) {
            other = bcVal[f];
          }
        }
        float snGrad = (deltaCoeffs[f] * (other - po));
        float phiTerm = __uint_as_float(0x00000000u);
        if (bcSnGradMask[f] == 2) {
          phiTerm = bcSnGrad[f];
        } else {
          if (bcSnGradMask[f] == 1) {
            snGrad = bcSnGrad[f];
          }
          float gradFace = (magSf[f] * snGrad);
          float denom = rAUf[f];
          float pFlux = ((paramsF[0] * (denom * gradFace)) + faceFluxCorrection[f]);
          if (denom != __uint_as_float(0x00000000u)) {
            phiTerm = ((phig[f] - pFlux) / denom);
          }
        }
        float ms = magSf[f];
        if (ms > __uint_as_float(0x00000000u)) {
          float nx = (sfX[f] / ms);
          float ny = (sfY[f] / ms);
          float nz = (sfZ[f] / ms);
          a00 = (a00 + (nx * sfX[f]));
          a01 = (a01 + (nx * sfY[f]));
          a02 = (a02 + (nx * sfZ[f]));
          a10 = (a10 + (ny * sfX[f]));
          a11 = (a11 + (ny * sfY[f]));
          a12 = (a12 + (ny * sfZ[f]));
          a20 = (a20 + (nz * sfX[f]));
          a21 = (a21 + (nz * sfY[f]));
          a22 = (a22 + (nz * sfZ[f]));
          bx = (bx + (nx * phiTerm));
          by = (by + (ny * phiTerm));
          bz = (bz + (nz * phiTerm));
        }
      }
    }
    float smallConst = __uint_as_float(0x33d6bf95u);
    float rootVSmall = __uint_as_float(0x1e3ce508u);
    float magSqr_xx = (a00 * a00);
    float magSqr_yy = (a11 * a11);
    float magSqr_zz = (a22 * a22);
    float threshold = (smallConst * ((magSqr_xx + magSqr_yy) + magSqr_zz));
    bool small_xx = (magSqr_xx < threshold);
    bool small_yy = (magSqr_yy < threshold);
    bool small_zz = (magSqr_zz < threshold);
    float w00 = a00;
    float w01 = a01;
    float w02 = a02;
    float w10 = a10;
    float w11 = a11;
    float w12 = a12;
    float w20 = a20;
    float w21 = a21;
    float w22 = a22;
    if (small_xx) {
      w00 = (w00 + __uint_as_float(0x3f800000u));
    }
    if (small_yy) {
      w11 = (w11 + __uint_as_float(0x3f800000u));
    }
    if (small_zz) {
      w22 = (w22 + __uint_as_float(0x3f800000u));
    }
    float det = (((((w00 * w11) * w22) + ((w01 * w12) * w20)) + ((w02 * w10) * w21)) - ((((w00 * w12) * w21) + ((w01 * w10) * w22)) + ((w02 * w11) * w20)));
    float rx = __uint_as_float(0x00000000u);
    float ry = __uint_as_float(0x00000000u);
    float rz = __uint_as_float(0x00000000u);
    if (abs(det) >= rootVSmall) {
      float invDet = (__uint_as_float(0x3f800000u) / det);
      float i00 = (((w11 * w22) - (w21 * w12)) * invDet);
      float i01 = (((w02 * w21) - (w01 * w22)) * invDet);
      float i02 = (((w01 * w12) - (w02 * w11)) * invDet);
      float i10 = (((w12 * w20) - (w10 * w22)) * invDet);
      float i11 = (((w00 * w22) - (w02 * w20)) * invDet);
      float i12 = (((w02 * w10) - (w00 * w12)) * invDet);
      float i20 = (((w10 * w21) - (w11 * w20)) * invDet);
      float i21 = (((w01 * w20) - (w00 * w21)) * invDet);
      float i22 = (((w00 * w11) - (w10 * w01)) * invDet);
      if (small_xx) {
        i00 = (i00 - __uint_as_float(0x3f800000u));
      }
      if (small_yy) {
        i11 = (i11 - __uint_as_float(0x3f800000u));
      }
      if (small_zz) {
        i22 = (i22 - __uint_as_float(0x3f800000u));
      }
      rx = (((i00 * bx) + (i01 * by)) + (i02 * bz));
      ry = (((i10 * bx) + (i11 * by)) + (i12 * bz));
      rz = (((i20 * bx) + (i21 * by)) + (i22 * bz));
    }
    float fac = rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hbx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (fac * rx));
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hby[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (fac * ry));
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hbz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (fac * rz));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_grad_alpha_cell = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_grad_alpha_cell(const float* alphaF, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* counts, float* gx, float* gy, float* gz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sx = __uint_as_float(0x00000000u);
    float sy = __uint_as_float(0x00000000u);
    float sz = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = (sign[k] * alphaF[f]);
      sx = (sx + (s * sfX[f]));
      sy = (sy + (s * sfY[f]));
      sz = (sz + (s * sfZ[f]));
    }
    float invV = (__uint_as_float(0x3f800000u) / vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    gx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sx * invV);
    gy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sy * invV);
    gz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sz * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_grad_alpha_cell_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_grad_alpha_cell_f64(const double* alphaF, const double* sfX, const double* sfY, const double* sfZ, const int* offsets, const int* faceIdx, const float* sign, const double* vol, const int* counts, double* gx, double* gy, double* gz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double sx = __longlong_as_double(0x0000000000000000ull);
    double sy = __longlong_as_double(0x0000000000000000ull);
    double sz = __longlong_as_double(0x0000000000000000ull);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double aV = ((double)(alphaF[f]));
      double s = (signV * aV);
      double sfXv = ((double)(sfX[f]));
      double sfYv = ((double)(sfY[f]));
      double sfZv = ((double)(sfZ[f]));
      sx = (sx + (s * sfXv));
      sy = (sy + (s * sfYv));
      sz = (sz + (s * sfZv));
    }
    double invV = (__longlong_as_double(0x3ff0000000000000ull) / ((double)(vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])));
    gx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sx * invV);
    gy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sy * invV);
    gz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sz * invV);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_n_hatf_all = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_n_hatf_all(const float* gx, const float* gy, const float* gz, const float* sfX, const float* sfY, const float* sfZ, const int* owner, const int* nei, const float* weights, const float* theta, const unsigned int* thetaMask, const float* paramsF, const int* counts, float* contactGrad, float* nHatf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    bool isBoundary = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]);
    float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float wm = (__uint_as_float(0x3f800000u) - w);
    float gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    float gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    float gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float inv = (__uint_as_float(0x3f800000u) / (magg + paramsF[0]));
    float nx = (gfx * inv);
    float ny = (gfy * inv);
    float nz = (gfz * inv);
    contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    if (thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u) {
      float sfx = sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float sfy = sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float sfz = sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > __uint_as_float(0x1e3ce508u)) {
        float nfx = (sfx / magSf);
        float nfy = (sfy / magSf);
        float nfz = (sfz / magSf);
        float a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        float a12 = a12raw;
        if (a12 < __uint_as_float(0xbf800000u)) {
          a12 = __uint_as_float(0xbf800000u);
        } else {
          if (a12 > __uint_as_float(0x3f800000u)) {
            a12 = __uint_as_float(0x3f800000u);
          }
        }
        float th = theta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float b1 = cos(th);
        float acosA12 = acos(a12);
        float b2 = cos((acosA12 - th));
        float det = (__uint_as_float(0x3f800000u) - (a12 * a12));
        if (det < __uint_as_float(0x358637bdu)) {
          det = __uint_as_float(0x358637bdu);
        }
        float a = ((b1 - (a12 * b2)) / det);
        float b = ((b2 - (a12 * b1)) / det);
        float nx2 = ((a * nfx) + (b * nx));
        float ny2 = ((a * nfy) + (b * ny));
        float nz2 = ((a * nfz) + (b * nz));
        float magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        float invN = (__uint_as_float(0x3f800000u) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        float dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((nx * sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) + (ny * sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) + (nz * sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_n_hatf_all_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_n_hatf_all_f64(const double* gx, const double* gy, const double* gz, const double* sfX, const double* sfY, const double* sfZ, const int* owner, const int* nei, const double* weights, const float* theta, const unsigned int* thetaMask, const double* paramsF, const int* counts, double* contactGrad, double* nHatf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    bool isBoundary = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]);
    double w = ((double)(weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    double wm = (__longlong_as_double(0x3ff0000000000000ull) - w);
    double gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    double gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    double gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    double magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    double inv = (__longlong_as_double(0x3ff0000000000000ull) / (magg + paramsF[0]));
    double nx = (gfx * inv);
    double ny = (gfy * inv);
    double nz = (gfz * inv);
    contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __longlong_as_double(0x0000000000000000ull);
    if (thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u) {
      double sfx = ((double)(sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      double sfy = ((double)(sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      double sfz = ((double)(sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      double magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > __longlong_as_double(0x3bc79ca10c924223ull)) {
        double nfx = (sfx / magSf);
        double nfy = (sfy / magSf);
        double nfz = (sfz / magSf);
        double a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        double a12 = a12raw;
        if (a12 < __longlong_as_double(0xbff0000000000000ull)) {
          a12 = __longlong_as_double(0xbff0000000000000ull);
        } else {
          if (a12 > __longlong_as_double(0x3ff0000000000000ull)) {
            a12 = __longlong_as_double(0x3ff0000000000000ull);
          }
        }
        double th = ((double)(theta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        double b1 = cos(th);
        double acosA12 = acos(a12);
        double b2 = cos((acosA12 - th));
        double det = (__longlong_as_double(0x3ff0000000000000ull) - (a12 * a12));
        if (det < __longlong_as_double(0x3eb0c6f7a0b5ed8dull)) {
          det = __longlong_as_double(0x3eb0c6f7a0b5ed8dull);
        }
        double a = ((b1 - (a12 * b2)) / det);
        double b = ((b2 - (a12 * b1)) / det);
        double nx2 = ((a * nfx) + (b * nx));
        double ny2 = ((a * nfy) + (b * ny));
        double nz2 = ((a * nfz) + (b * nz));
        double magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        double invN = (__longlong_as_double(0x3ff0000000000000ull) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        double dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((nx * ((double)(sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]))) + (ny * ((double)(sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])))) + (nz * ((double)(sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]))));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_stf_all_faces = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_stf_all_faces(const float* divN, const float* alpha, const float* alphaF, const float* delta, const int* owner, const int* nei, const float* weights, const float* contactGrad, const unsigned int* thetaMask, const float* paramsF, const int* counts, float* stf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float wm = (__uint_as_float(0x3f800000u) - w);
    float divInterp = ((w * divN[o]) + (wm * divN[n]));
    float Kf = (__uint_as_float(0xbf800000u) * divInterp);
    float snGrad = __uint_as_float(0x00000000u);
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2]) {
      snGrad = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (alpha[n] - alpha[o]));
    } else {
      if (thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u) {
        snGrad = contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      } else {
        snGrad = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - alpha[o]));
      }
    }
    stf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((paramsF[0] * Kf) * snGrad);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_stf_all_faces_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_stf_all_faces_f64(const double* divN, const float* alpha, const float* alphaF, const float* delta, const int* owner, const int* nei, const float* weights, const double* contactGrad, const unsigned int* thetaMask, const float* paramsF, const int* counts, double* stf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    double w = ((double)(weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    double wm = (__longlong_as_double(0x3ff0000000000000ull) - w);
    double divInterp = ((w * divN[o]) + (wm * divN[n]));
    double Kf = (-divInterp);
    double snGrad = __longlong_as_double(0x0000000000000000ull);
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2]) {
      snGrad = (((double)(delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) * (((double)(alpha[n])) - ((double)(alpha[o]))));
    } else {
      if (thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u) {
        snGrad = contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      } else {
        snGrad = (((double)(delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) * (((double)(alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) - ((double)(alpha[o]))));
      }
    }
    stf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((((double)(paramsF[0])) * Kf) * snGrad);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_k_omega_s_s_t_blending = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_k_omega_s_s_t_blending(const float* k, const float* omega, const float* nut, const float* nu, const float* y, const float* gradKx, const float* gradKy, const float* gradKz, const float* gradOmegax, const float* gradOmegay, const float* gradOmegaz, const int* counts, float* outF1, float* outF23, float* outCDkOmega) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float omC = omega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nutC = nut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nuC = nu[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float yC = y[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gKx = gradKx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gKy = gradKy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gKz = gradKz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gOx = gradOmegax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gOy = gradOmegay[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gOz = gradOmegaz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float dotGrad = (((gKx * gOx) + (gKy * gOy)) + (gKz * gOz));
    float omSafe = (((omC > __uint_as_float(0x2edbe6ffu))) ? (omC) : (__uint_as_float(0x2edbe6ffu)));
    float cdkw = ((__uint_as_float(0x3fdb22d1u) * dotGrad) / omSafe);
    float cdkwPlus = (((cdkw > __uint_as_float(0x2edbe6ffu))) ? (cdkw) : (__uint_as_float(0x2edbe6ffu)));
    float ySq = (yC * yC);
    float ySqSafe = (((ySq > __uint_as_float(0x1e3ce508u))) ? (ySq) : (__uint_as_float(0x1e3ce508u)));
    float kSafe = (((kC > __uint_as_float(0x00000000u))) ? (kC) : (__uint_as_float(0x00000000u)));
    float sqrtK = sqrt(kSafe);
    float denom1 = (__uint_as_float(0x3db851ecu) * (omSafe * yC));
    float denom1Safe = (((denom1 > __uint_as_float(0x2edbe6ffu))) ? (denom1) : (__uint_as_float(0x2edbe6ffu)));
    float term1 = (sqrtK / denom1Safe);
    float denom2 = (ySqSafe * omSafe);
    float term2 = ((__uint_as_float(0x43fa0000u) * nuC) / denom2);
    float maxT12 = (((term1 >= term2)) ? (term1) : (term2));
    float denom3 = (cdkwPlus * ySqSafe);
    float denom3Safe = (((denom3 > __uint_as_float(0x2edbe6ffu))) ? (denom3) : (__uint_as_float(0x2edbe6ffu)));
    float term3 = ((__uint_as_float(0x405b22d1u) * kC) / denom3Safe);
    float minT3 = (((maxT12 <= term3)) ? (maxT12) : (term3));
    float arg1 = (((minT3 <= __uint_as_float(0x41200000u))) ? (minT3) : (__uint_as_float(0x41200000u)));
    float arg1sq = (arg1 * arg1);
    float arg1p4 = (arg1sq * arg1sq);
    float arg1p4Safe = (((arg1p4 <= __uint_as_float(0x42200000u))) ? (arg1p4) : (__uint_as_float(0x42200000u)));
    float f1Val = tanh(arg1p4Safe);
    float term2a = ((__uint_as_float(0x40000000u) * sqrtK) / denom1Safe);
    float maxF2 = (((term2a >= term2)) ? (term2a) : (term2));
    float arg2 = (((maxF2 <= __uint_as_float(0x42c80000u))) ? (maxF2) : (__uint_as_float(0x42c80000u)));
    float arg2sq = (arg2 * arg2);
    float arg2sqSafe = (((arg2sq <= __uint_as_float(0x42200000u))) ? (arg2sq) : (__uint_as_float(0x42200000u)));
    float f23Val = tanh(arg2sqSafe);
    outF1[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = f1Val;
    outF23[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = f23Val;
    outCDkOmega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = cdkw;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_k_omega_s_s_t_nut_compute = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_k_omega_s_s_t_nut_compute(const float* kVals, const float* omegaVals, const float* f23Vals, const float* s2Vals, const unsigned int* counts, const float* params, float* outNut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float a1 = params[0];
    float b1 = params[1];
    float k = kVals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float om = omegaVals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float f23 = f23Vals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float s2raw = s2Vals[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float denomA = (a1 * om);
    float denomB = ((b1 * f23) * sqrt((s2raw * ((float)((s2raw > __uint_as_float(0x00000000u)))))));
    float num = (a1 * k);
    if (denomA > denomB) {
      if (denomA > __uint_as_float(0x00000000u)) {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (num / denomA);
      } else {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
      }
    } else {
      if (denomB > __uint_as_float(0x00000000u)) {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (num / denomB);
      } else {
        outNut[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_k_omega_s_s_t_sources = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_k_omega_s_s_t_sources(const float* k, const float* omega, const float* nut, const float* nu, const float* alpha, const float* rho, const float* F1, const float* F23, const float* CDkOmega, const float* S2, const float* divU, const int* counts, float* outOmegaSp, float* outOmegaSuSp1, float* outOmegaSuSp2, float* outOmegaExpl, float* outKSp, float* outKSuSp, float* outKExpl, float* outDomegaEff, float* outDkEff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float omC = omega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nutC = nut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nuC = nu[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float alphaC = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float rhoC = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float f1C = F1[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float f23C = F23[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float cdkwC = CDkOmega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float s2C = S2[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float divUC = divU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float arC = (alphaC * rhoC);
    float gammaC = ((f1C * __uint_as_float(0x3decbfb1u)) + __uint_as_float(0x3ee147aeu));
    float betaC = (__uint_as_float(0x3da9930cu) - (f1C * __uint_as_float(0x3bff9724u)));
    float alphaKC = (__uint_as_float(0x3f800000u) - (f1C * __uint_as_float(0x3e19999au)));
    float alphaOmC = (__uint_as_float(0x3f5b22d1u) - (f1C * __uint_as_float(0x3eb645a2u)));
    float s2Safe = (((s2C > __uint_as_float(0x00000000u))) ? (s2C) : (__uint_as_float(0x00000000u)));
    float sqrtS2 = sqrt(s2Safe);
    float a1om = (__uint_as_float(0x3e9eb852u) * omC);
    float f23s = (f23C * sqrtS2);
    float maxTerm = (((a1om >= f23s)) ? (a1om) : (f23s));
    float limRhs = (__uint_as_float(0x4039ce07u) * (omC * maxTerm));
    float GbyNu = (((s2C <= limRhs)) ? (s2C) : (limRhs));
    float nutS2 = (nutC * s2C);
    float c1bsKOm = (__uint_as_float(0x3f666666u) * (kC * omC));
    float pk = (((nutS2 <= c1bsKOm)) ? (nutS2) : (c1bsKOm));
    float omSp = (__uint_as_float(0x00000000u) - (arC * (betaC * omC)));
    float omSuSp1 = (__uint_as_float(0x00000000u) - (__uint_as_float(0x3f2aacdau) * (arC * (gammaC * divUC))));
    float omSafe = (((omC > __uint_as_float(0x26901d7du))) ? (omC) : (__uint_as_float(0x26901d7du)));
    float f1m1 = (f1C - __uint_as_float(0x3f800000u));
    float omSuSp2 = (__uint_as_float(0x00000000u) - ((arC * (f1m1 * cdkwC)) / omSafe));
    float omExpl = (arC * (gammaC * GbyNu));
    float kSp = (__uint_as_float(0x00000000u) - (arC * (__uint_as_float(0x3db851ecu) * omC)));
    float kSuSp = (__uint_as_float(0x00000000u) - (__uint_as_float(0x3f2aacdau) * (arC * divUC)));
    float kExpl = (arC * pk);
    float domEff = (arC * ((alphaOmC * nutC) + nuC));
    float dkEff = (arC * ((alphaKC * nutC) + nuC));
    outOmegaSp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omSp;
    outOmegaSuSp1[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omSuSp1;
    outOmegaSuSp2[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omSuSp2;
    outOmegaExpl[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omExpl;
    outKSp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kSp;
    outKSuSp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kSuSp;
    outKExpl[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kExpl;
    outDomegaEff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = domEff;
    outDkEff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = dkEff;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mu_eff_from_rho_nu_eff = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mu_eff_from_rho_nu_eff(const float* rho, const float* nuEff, const int* params, float* outMuEff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float r = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nu = nuEff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outMuEff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * nu);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_mu_eff_from_rho_nut_and_nu_mix(const float* rho, const float* nut, const float* nuMix, const unsigned int* params, float* outMu) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    outMu[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (rho[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * (nuMix[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + nut[(int)(blockIdx.x * blockDim.x + threadIdx.x)]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_turbulence_add_explicit_source_device = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_turbulence_add_explicit_source_device(const float* explicitSrc, const float* vol, const unsigned int* counts, float* outVals) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    outVals[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (explicitSrc[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * vol[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_turbulence_dirichlet_cell_source = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_turbulence_dirichlet_cell_source(const int* constrainedMask, const float* constrainedValue, const float* diag, const int* nCells, float* source) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= nCells[0]) {
    return;
  } else {
    int mask = constrainedMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (mask == 0) {
      return;
    } else {
      source[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (diag[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * constrainedValue[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_turbulence_dirichlet_face_elimination = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_turbulence_dirichlet_face_elimination(const int* owner, const int* neighbour, const int* constrainedMask, const float* constrainedValue, const int* nIntFaces, float* upper, float* lower, float* source) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= nIntFaces[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int oC = constrainedMask[o];
    int nC = constrainedMask[n];
    if ((oC | nC) == 0) {
      return;
    } else {
      float up = upper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float lo = lower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (oC != 0) {
        float cv = constrainedValue[o];
        atomicAdd(&source[n], (-(lo * cv)));
      }
      if (nC != 0) {
        float cv = constrainedValue[n];
        atomicAdd(&source[o], (-(up * cv)));
      }
      upper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      lower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_two_mag_sqr_symm_grad_u = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_two_mag_sqr_symm_grad_u(const float* ufX, const float* ufY, const float* ufZ, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* counts, float* outS2) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float g00 = __uint_as_float(0x00000000u);
    float g01 = __uint_as_float(0x00000000u);
    float g02 = __uint_as_float(0x00000000u);
    float g10 = __uint_as_float(0x00000000u);
    float g11 = __uint_as_float(0x00000000u);
    float g12 = __uint_as_float(0x00000000u);
    float g20 = __uint_as_float(0x00000000u);
    float g21 = __uint_as_float(0x00000000u);
    float g22 = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float sx = (s * sfX[f]);
      float sy = (s * sfY[f]);
      float sz = (s * sfZ[f]);
      float ux = ufX[f];
      float uy = ufY[f];
      float uz = ufZ[f];
      g00 = (g00 + (sx * ux));
      g01 = (g01 + (sx * uy));
      g02 = (g02 + (sx * uz));
      g10 = (g10 + (sy * ux));
      g11 = (g11 + (sy * uy));
      g12 = (g12 + (sy * uz));
      g20 = (g20 + (sz * ux));
      g21 = (g21 + (sz * uy));
      g22 = (g22 + (sz * uz));
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    g00 = (g00 * invV);
    g01 = (g01 * invV);
    g02 = (g02 * invV);
    g10 = (g10 * invV);
    g11 = (g11 * invV);
    g12 = (g12 * invV);
    g20 = (g20 * invV);
    g21 = (g21 * invV);
    g22 = (g22 * invV);
    float s00 = g00;
    float s11 = g11;
    float s22 = g22;
    float s01 = (__uint_as_float(0x3f000000u) * (g01 + g10));
    float s02 = (__uint_as_float(0x3f000000u) * (g02 + g20));
    float s12 = (__uint_as_float(0x3f000000u) * (g12 + g21));
    float d0 = (s00 * s00);
    float d1 = (s11 * s11);
    float d2 = (s22 * s22);
    float o0 = (s01 * s01);
    float o1 = (s02 * s02);
    float o2 = (s12 * s12);
    float diag = ((d0 + d1) + d2);
    float off = ((o0 + o1) + o2);
    outS2[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((__uint_as_float(0x40000000u) * diag) + (__uint_as_float(0x40800000u) * off));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_attention_row_softmax = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_attention_row_softmax(const float* scores, const int* numQ, const int* numK, float* weights) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int qi = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nq = numQ[0];
  int nk = numK[0];
  if (qi < nq) {
    int rowBase = (qi * nk);
    float maxScore = __uint_as_float(0xf149f2cau);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      maxScore = (((s > maxScore)) ? (s) : (maxScore));
    }
    float sumExp = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      sumExp = (sumExp + e);
    }
    float denom = (sumExp + __uint_as_float(0x3089705fu));
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      float w = (e / denom);
      weights[idx] = w;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_attention_value_projection = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_attention_value_projection(const float* weights, const float* v, const int* numQ, const int* numK, const int* headDim, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * hd);
  if (gid < total) {
    int qi = (gid / hd);
    int d = (gid % hd);
    float acc = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int wIdx = ((qi * nk) + j);
      int vIdx = ((j * hd) + d);
      float w = weights[wIdx];
      float vv = v[vIdx];
      acc = (acc + (w * vv));
    }
    _out[gid] = acc;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gather_mean = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gather_mean(const float* features, const int* neighbors, const int* neighborCounts, const int* numNodes, const int* featureDim, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nNodes = numNodes[0];
  if (gid < nNodes) {
    int cnt = neighborCounts[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      float acc = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(cnt); ++j) {
        int nbrFlat = ((gid * 8) + j);
        int nbr = neighbors[nbrFlat];
        int featFlat = ((nbr * fd) + f);
        float v = features[featFlat];
        acc = (acc + v);
      }
      float cntF = ((float)(cnt));
      float mean = (((cnt > 0)) ? ((acc / cntF)) : (__uint_as_float(0x00000000u)));
      int outFlat = ((gid * fd) + f);
      _out[outFlat] = mean;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_layer_norm_forward = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_layer_norm_forward(const float* x, const int* numRows, const float* gamma, const float* beta, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nRows = numRows[0];
  if (row < nRows) {
    int hd = 16;
    int rowBase = (row * hd);
    float sum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      sum = (sum + xv);
    }
    float hdF = ((float)(hd));
    float mean = (sum / hdF);
    float varSum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float diff = (xv - mean);
      varSum = (varSum + (diff * diff));
    }
    float var = (varSum / hdF);
    float invStd = sqrt((__uint_as_float(0x3f800000u) / (var + __uint_as_float(0x3727c5acu))));
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float xn = ((xv - mean) * invStd);
      float g = gamma[j];
      float b = beta[j];
      float yv = ((xn * g) + b);
      y[idx] = yv;
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_radius_count = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_radius_count(const float* query, const float* ref, const int* numQuery, const int* numRef, const float* radiusIn, int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nq = numQuery[0];
  int nr = numRef[0];
  if (qid < nq) {
    float qx = query[((qid)*3 + 0)];
    float qy = query[((qid)*3 + 1)];
    float qz = query[((qid)*3 + 2)];
    float rIn = radiusIn[0];
    float rCmp = rIn;
    int cnt = 0;
    for (int rid = 0; rid < (int)(nr); ++rid) {
      float rx = ref[((rid)*3 + 0)];
      float ry = ref[((rid)*3 + 1)];
      float rz = ref[((rid)*3 + 2)];
      float dx = (qx - rx);
      float dy = (qy - ry);
      float dz = (qz - rz);
      float d2 = (((dx * dx) + (dy * dy)) + (dz * dz));
      if (d2 <= rCmp) {
        cnt = (cnt + 1);
      }
    }
    counts[qid] = cnt;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scaled_dot_product = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scaled_dot_product(const float* q, const float* k, const int* numQ, const int* numK, const int* headDim, float* scores) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * nk);
  if (gid < total) {
    int qi = (gid / nk);
    int kj = (gid % nk);
    float acc = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      int qFlat = ((qi * hd) + d);
      int kFlat = ((kj * hd) + d);
      float qv = q[qFlat];
      float kv = k[kFlat];
      acc = (acc + (qv * kv));
    }
    scores[gid] = (acc * __uint_as_float(0x3e800000u));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_scatter_add_atomic = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_scatter_add_atomic(const int* values, const int* indices, const int* numEdges, const int* featureDim, int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nEdges = numEdges[0];
  if (gid < nEdges) {
    int dstIdx = indices[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      int valFlat = ((gid * fd) + f);
      int dstFlat = ((dstIdx * fd) + f);
      int v = values[valFlat];
      int _ignored = atomicAdd(&(dst[dstFlat]), v);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_spectral_complex_mul = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_spectral_complex_mul(const float* x, const float* w, const int* n, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int count = n[0];
  if (gid < count) {
    float xr = x[((gid)*2 + 0)];
    float xi = x[((gid)*2 + 1)];
    float wr = w[((gid)*2 + 0)];
    float wi = w[((gid)*2 + 1)];
    float yr = ((xr * wr) - (xi * wi));
    float yi = ((xr * wi) + (xi * wr));
    y[((gid)*2 + 0)] = yr;
    y[((gid)*2 + 1)] = yi;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_spectral_conv1d_mode_sliced = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_spectral_conv1d_mode_sliced(const float* xFt, const float* wPos, const float* wNeg, const int* numBatches, const int* ftW, float* yFt) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nb = numBatches[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 8;
  int total = ((nb * outCh) * ftw);
  if (gid < total) {
    int boc = (gid / ftw);
    int w = (gid % ftw);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    float accR = __uint_as_float(0x00000000u);
    float accI = __uint_as_float(0x00000000u);
    if (w < m1) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((b * inCh) + ic) * ftw) + w);
        int wBase = ((((oc * inCh) + ic) * m1) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPos[((wBase)*2 + 0)];
        float wi = wPos[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      int negStart = (ftw - m1);
      if (w >= negStart) {
        int wLocal = (w - negStart);
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((b * inCh) + ic) * ftw) + w);
          int wBase = ((((oc * inCh) + ic) * m1) + wLocal);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNeg[((wBase)*2 + 0)];
          float wi = wNeg[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_spectral_conv2d_mode_sliced = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_spectral_conv2d_mode_sliced(const float* xFt, const float* wPos, const float* wNeg, const int* numBatches, const int* ftH, const int* ftW, float* yFt) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nb = numBatches[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 8;
  int m2 = 8;
  int plane = (fth * ftw);
  int total = ((nb * outCh) * plane);
  if (gid < total) {
    int boc = (gid / plane);
    int hwFlat = (gid % plane);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    float accR = __uint_as_float(0x00000000u);
    float accI = __uint_as_float(0x00000000u);
    if ((h < m1) && (w < m2)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((b * inCh) + ic) * fth) + h) * ftw) + w);
        int wBase = ((((((oc * inCh) + ic) * m1) + h) * m2) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPos[((wBase)*2 + 0)];
        float wi = wPos[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      int negStart = (fth - m1);
      if ((h >= negStart) && (w < m2)) {
        int hLocal = (h - negStart);
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((b * inCh) + ic) * fth) + h) * ftw) + w);
          int wBase = ((((((oc * inCh) + ic) * m1) + hLocal) * m2) + w);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNeg[((wBase)*2 + 0)];
          float wi = wNeg[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_spectral_conv3d_mode_sliced = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_spectral_conv3d_mode_sliced(const float* xFt, const float* wPP, const float* wNP, const float* wPN, const float* wNN, const int* numBatches, const int* ftD, const int* ftH, const int* ftW, float* yFt) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nb = numBatches[0];
  int ftd = ftD[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 4;
  int m2 = 4;
  int m3 = 4;
  int plane = ((ftd * fth) * ftw);
  int total = ((nb * outCh) * plane);
  if (gid < total) {
    int boc = (gid / plane);
    int dhwFlat = (gid % plane);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int d = (dhwFlat / (fth * ftw));
    int hwFlat = (dhwFlat % (fth * ftw));
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    int negStartD = (ftd - m1);
    int negStartH = (fth - m2);
    float accR = __uint_as_float(0x00000000u);
    float accI = __uint_as_float(0x00000000u);
    if (((d < m1) && (h < m2)) && (w < m3)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
        int wBase = ((((((((oc * inCh) + ic) * m1) + d) * m2) + h) * m3) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPP[((wBase)*2 + 0)];
        float wi = wPP[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      if (((d >= negStartD) && (h < m2)) && (w < m3)) {
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
          int wBase = ((((((((oc * inCh) + ic) * m1) + (d - negStartD)) * m2) + h) * m3) + w);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNP[((wBase)*2 + 0)];
          float wi = wNP[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      } else {
        if (((d < m1) && (h >= negStartH)) && (w < m3)) {
          for (int ic = 0; ic < (int)(inCh); ++ic) {
            int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
            int wBase = ((((((((oc * inCh) + ic) * m1) + d) * m2) + (h - negStartH)) * m3) + w);
            float xr = xFt[((xBase)*2 + 0)];
            float xi = xFt[((xBase)*2 + 1)];
            float wr = wPN[((wBase)*2 + 0)];
            float wi = wPN[((wBase)*2 + 1)];
            accR = (accR + ((xr * wr) - (xi * wi)));
            accI = (accI + ((xr * wi) + (xi * wr)));
          }
        } else {
          if (((d >= negStartD) && (h >= negStartH)) && (w < m3)) {
            for (int ic = 0; ic < (int)(inCh); ++ic) {
              int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
              int wBase = ((((((((oc * inCh) + ic) * m1) + (d - negStartD)) * m2) + (h - negStartH)) * m3) + w);
              float xr = xFt[((xBase)*2 + 0)];
              float xi = xFt[((xBase)*2 + 1)];
              float wr = wNN[((wBase)*2 + 0)];
              float wi = wNN[((wBase)*2 + 1)];
              accR = (accR + ((xr * wr) - (xi * wi)));
              accI = (accI + ((xr * wi) + (xi * wr)));
            }
          }
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_spectral_conv4d_mode_sliced = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_spectral_conv4d_mode_sliced(const float* xFt, const float* wPPP, const float* wNPP, const float* wPNP, const float* wNNP, const float* wPPN, const float* wNPN, const float* wPNN, const float* wNNN, const int* numBatches, const int* ftT, const int* ftD, const int* ftH, const int* ftW, float* yFt) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nb = numBatches[0];
  int ftt = ftT[0];
  int ftd = ftD[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 2;
  int m2 = 2;
  int m3 = 2;
  int m4 = 2;
  int hyper = (((ftt * ftd) * fth) * ftw);
  int total = ((nb * outCh) * hyper);
  if (gid < total) {
    int boc = (gid / hyper);
    int tdhwFlat = (gid % hyper);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int t = (tdhwFlat / ((ftd * fth) * ftw));
    int dhwFlat = (tdhwFlat % ((ftd * fth) * ftw));
    int d = (dhwFlat / (fth * ftw));
    int hwFlat = (dhwFlat % (fth * ftw));
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    int negStartT = (ftt - m1);
    int negStartD = (ftd - m2);
    int negStartH = (fth - m3);
    float accR = __uint_as_float(0x00000000u);
    float accI = __uint_as_float(0x00000000u);
    if ((((t < m1) && (d < m2)) && (h < m3)) && (w < m4)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
        int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + d) * m3) + h) * m4) + w) * 1) + 0);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPPP[((wBase)*2 + 0)];
        float wi = wPPP[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      if ((((t >= negStartT) && (d < m2)) && (h < m3)) && (w < m4)) {
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
          int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + d) * m3) + h) * m4) + w) * 1) + 0);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNPP[((wBase)*2 + 0)];
          float wi = wNPP[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      } else {
        if ((((t < m1) && (d >= negStartD)) && (h < m3)) && (w < m4)) {
          for (int ic = 0; ic < (int)(inCh); ++ic) {
            int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
            int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + (d - negStartD)) * m3) + h) * m4) + w) * 1) + 0);
            float xr = xFt[((xBase)*2 + 0)];
            float xi = xFt[((xBase)*2 + 1)];
            float wr = wPNP[((wBase)*2 + 0)];
            float wi = wPNP[((wBase)*2 + 1)];
            accR = (accR + ((xr * wr) - (xi * wi)));
            accI = (accI + ((xr * wi) + (xi * wr)));
          }
        } else {
          if ((((t >= negStartT) && (d >= negStartD)) && (h < m3)) && (w < m4)) {
            for (int ic = 0; ic < (int)(inCh); ++ic) {
              int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
              int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + (d - negStartD)) * m3) + h) * m4) + w) * 1) + 0);
              float xr = xFt[((xBase)*2 + 0)];
              float xi = xFt[((xBase)*2 + 1)];
              float wr = wNNP[((wBase)*2 + 0)];
              float wi = wNNP[((wBase)*2 + 1)];
              accR = (accR + ((xr * wr) - (xi * wi)));
              accI = (accI + ((xr * wi) + (xi * wr)));
            }
          } else {
            if ((((t < m1) && (d < m2)) && (h >= negStartH)) && (w < m4)) {
              for (int ic = 0; ic < (int)(inCh); ++ic) {
                int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + d) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                float xr = xFt[((xBase)*2 + 0)];
                float xi = xFt[((xBase)*2 + 1)];
                float wr = wPPN[((wBase)*2 + 0)];
                float wi = wPPN[((wBase)*2 + 1)];
                accR = (accR + ((xr * wr) - (xi * wi)));
                accI = (accI + ((xr * wi) + (xi * wr)));
              }
            } else {
              if ((((t >= negStartT) && (d < m2)) && (h >= negStartH)) && (w < m4)) {
                for (int ic = 0; ic < (int)(inCh); ++ic) {
                  int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                  int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + d) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                  float xr = xFt[((xBase)*2 + 0)];
                  float xi = xFt[((xBase)*2 + 1)];
                  float wr = wNPN[((wBase)*2 + 0)];
                  float wi = wNPN[((wBase)*2 + 1)];
                  accR = (accR + ((xr * wr) - (xi * wi)));
                  accI = (accI + ((xr * wi) + (xi * wr)));
                }
              } else {
                if ((((t < m1) && (d >= negStartD)) && (h >= negStartH)) && (w < m4)) {
                  for (int ic = 0; ic < (int)(inCh); ++ic) {
                    int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                    int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + (d - negStartD)) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                    float xr = xFt[((xBase)*2 + 0)];
                    float xi = xFt[((xBase)*2 + 1)];
                    float wr = wPNN[((wBase)*2 + 0)];
                    float wi = wPNN[((wBase)*2 + 1)];
                    accR = (accR + ((xr * wr) - (xi * wi)));
                    accI = (accI + ((xr * wi) + (xi * wr)));
                  }
                } else {
                  if ((((t >= negStartT) && (d >= negStartD)) && (h >= negStartH)) && (w < m4)) {
                    for (int ic = 0; ic < (int)(inCh); ++ic) {
                      int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                      int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + (d - negStartD)) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                      float xr = xFt[((xBase)*2 + 0)];
                      float xi = xFt[((xBase)*2 + 1)];
                      float wr = wNNN[((wBase)*2 + 0)];
                      float wi = wNNN[((wBase)*2 + 1)];
                      accR = (accR + ((xr * wr) - (xi * wi)));
                      accI = (accI + ((xr * wi) + (xi * wr)));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_compute_face_area_pair_weights_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_compute_face_area_pair_weights_f32(const float* sfX, const float* sfY, const float* sfZ, const float* magSf, const unsigned int* nFaces, float* faceWeight) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int f = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFaces[0]));
  if (f < N) {
    float sx = sfX[f];
    float sy = sfY[f];
    float sz = sfZ[f];
    float msf = magSf[f];
    float invSqrt = (__uint_as_float(0x3f800000u) / sqrt(msf));
    float wx = (sx * invSqrt);
    float wy = ((sy * __uint_as_float(0x3f8147aeu)) * invSqrt);
    float wz = ((sz * __uint_as_float(0x3f828f5cu)) * invSqrt);
    float mag2 = (((wx * wx) + (wy * wy)) + (wz * wz));
    faceWeight[f] = sqrt(mag2);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_restrict_by_agglomeration_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_restrict_by_agglomeration_f32(const float* fineField, const unsigned int* fineToCoarse, const unsigned int* nFineCells, float* coarseField) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFineCells[0]));
  if (i < N) {
    int coarseIdx = ((int)(fineToCoarse[i]));
    float val = fineField[i];
    float _a = atomicAdd(&(coarseField[coarseIdx]), val);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_restrict_by_agglomeration_gather_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_restrict_by_agglomeration_gather_f32(const float* fineField, const unsigned int* coarseToFineOffsets, const unsigned int* coarseToFineList, const unsigned int* nCoarseCells, float* coarseField) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCoarseCells[0]));
  if (c < N) {
    int begin = ((int)(coarseToFineOffsets[c]));
    int end = ((int)(coarseToFineOffsets[(c + 1)]));
    float sum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int fineIdx = ((int)(coarseToFineList[pos]));
      float v = fineField[fineIdx];
      sum = (sum + v);
    }
    coarseField[c] = sum;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_prolongate_by_agglomeration_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_prolongate_by_agglomeration_f32(const float* coarseField, const unsigned int* fineToCoarse, const unsigned int* nFineCells, float* fineField) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFineCells[0]));
  if (i < N) {
    int coarseIdx = ((int)(fineToCoarse[i]));
    float val = coarseField[coarseIdx];
    float cur = fineField[i];
    fineField[i] = (cur + val);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_diag_inv_apply_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_diag_inv_apply_f32(const float* r, const float* diagInv, const unsigned int* n, float* z) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (i < N) {
    float ri = r[i];
    float di = diagInv[i];
    z[i] = (di * ri);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_csr_sp_m_v_mixed_f32_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_csr_sp_m_v_mixed_f32_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* x, const unsigned int* nRows, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int start = ((int)(rowPtr[row]));
    int end = ((int)(rowPtr[(row + 1)]));
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((end - start)); ++j) {
      int idx = (start + j);
      float val = values[idx];
      int col = ((int)(colIdx[idx]));
      float xi = x[col];
      acc = (acc + (((double)(val)) * ((double)(xi))));
    }
    y[row] = ((float)(acc));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_chebyshev_sweep_mixed_f32_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_chebyshev_sweep_mixed_f32_f64(const float* r, const float* d, const double* coeffA, const double* coeffB, const unsigned int* n, float* x, float* dNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (i < N) {
    double cA = coeffA[0];
    double cB = coeffB[0];
    double ri = ((double)(r[i]));
    double di = ((double)(d[i]));
    double xi = ((double)(x[i]));
    double corr = ((cA * ri) + (cB * di));
    x[i] = ((float)((xi + corr)));
    dNew[i] = ((float)(corr));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_ell_sp_m_v_mixed_f32_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_ell_sp_m_v_mixed_f32_f64(const unsigned int* colIdx, const float* values, const float* x, const unsigned int* nRows, const unsigned int* maxWidth, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int W = ((int)(maxWidth[0]));
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int col = 0; col < (int)(W); ++col) {
      int idx = (row + (col * N));
      int c = ((int)(colIdx[idx]));
      float v = values[idx];
      float xi = x[c];
      acc = (acc + (((double)(v)) * ((double)(xi))));
    }
    y[row] = ((float)(acc));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_chebyshev_smooth_post_fused_mixed = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_chebyshev_smooth_post_fused_mixed(const float* Az, const float* r, const float* diagInv, const float* dPrev, const double* coeffA, const double* coeffB, const unsigned int* n, float* z, float* dNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if (i < N) {
    double cA = coeffA[0];
    double cB = coeffB[0];
    double az_i = ((double)(Az[i]));
    double r_i = ((double)(r[i]));
    double di_i = ((double)(diagInv[i]));
    double dp_i = ((double)(dPrev[i]));
    double z_i = ((double)(z[i]));
    double res = (r_i - az_i);
    double dinvr = (di_i * res);
    double corr = ((cA * dinvr) + (cB * dp_i));
    z[i] = ((float)((z_i + corr)));
    dNew[i] = ((float)(corr));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_proc_pack_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_proc_pack_f32(const float* x, const unsigned int* localCells, const unsigned int* count, float* sendVals) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(count[0]));
  if (i < N) {
    int cell = ((int)(localCells[i]));
    float v = x[cell];
    sendVals[i] = v;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_proc_add_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_proc_add_f32(const unsigned int* localCells, const float* coeffs, const float* recvVals, const unsigned int* count, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(count[0]));
  if (i < N) {
    int cell = ((int)(localCells[i]));
    float co = coeffs[i];
    float rv = recvVals[i];
    float contrib = (co * rv);
    float _a = atomicAdd(&(y[cell]), contrib);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_gamg_proc_add_gather_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_gamg_proc_add_gather_f32(const unsigned int* entryOffsets, const unsigned int* entryIndices, const float* coeffs, const float* recvVals, const unsigned int* nCells, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCells[0]));
  if (c < N) {
    int begin = ((int)(entryOffsets[c]));
    int end = ((int)(entryOffsets[(c + 1)]));
    float acc = y[c];
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int entry = ((int)(entryIndices[pos]));
      float co = coeffs[entry];
      float rv = recvVals[entry];
      acc = (acc + (co * rv));
    }
    y[c] = acc;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_boundary_force_partials[] = {
  { "pAllFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfComponent", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_boundary_force_partials = {
  1,
  "navatala_cfd_boundary_force_partials",
  "cuda",
  "navatala_cfd_boundary_force_partials",
  "kernel:cuda:navatala_cfd_boundary_force_partials",
  "abi-r1:cuda:navatala_cfd_boundary_force_partials",
  "abi-r1:cuda:navatala_cfd_boundary_force_partials",
  4,
  kAbiArgs_cuda_navatala_cfd_boundary_force_partials
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pack_owner_values[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSend", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pack_owner_values = {
  1,
  "navatala_cfd_pack_owner_values",
  "cuda",
  "navatala_cfd_pack_owner_values",
  "kernel:cuda:navatala_cfd_pack_owner_values",
  "abi-r1:cuda:navatala_cfd_pack_owner_values",
  "abi-r1:cuda:navatala_cfd_pack_owner_values",
  5,
  kAbiArgs_cuda_navatala_cfd_pack_owner_values
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scatter_values_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recv", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bcVal", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scatter_values_and_mask = {
  1,
  "navatala_cfd_scatter_values_and_mask",
  "cuda",
  "navatala_cfd_scatter_values_and_mask",
  "kernel:cuda:navatala_cfd_scatter_values_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_values_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_values_and_mask",
  5,
  kAbiArgs_cuda_navatala_cfd_scatter_values_and_mask
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pack_mu_grad_owner[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYX", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYY", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYZ", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZX", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZY", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZZ", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pack_mu_grad_owner = {
  1,
  "navatala_cfd_pack_mu_grad_owner",
  "cuda",
  "navatala_cfd_pack_mu_grad_owner",
  "kernel:cuda:navatala_cfd_pack_mu_grad_owner",
  "abi-r1:cuda:navatala_cfd_pack_mu_grad_owner",
  "abi-r1:cuda:navatala_cfd_pack_mu_grad_owner",
  14,
  kAbiArgs_cuda_navatala_cfd_pack_mu_grad_owner
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pack_owner_rho_vec3[] = {
  { "rhoCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pack_owner_rho_vec3 = {
  1,
  "navatala_cfd_pack_owner_rho_vec3",
  "cuda",
  "navatala_cfd_pack_owner_rho_vec3",
  "kernel:cuda:navatala_cfd_pack_owner_rho_vec3",
  "abi-r1:cuda:navatala_cfd_pack_owner_rho_vec3",
  "abi-r1:cuda:navatala_cfd_pack_owner_rho_vec3",
  8,
  kAbiArgs_cuda_navatala_cfd_pack_owner_rho_vec3
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pack_owner_vec3[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pack_owner_vec3 = {
  1,
  "navatala_cfd_pack_owner_vec3",
  "cuda",
  "navatala_cfd_pack_owner_vec3",
  "kernel:cuda:navatala_cfd_pack_owner_vec3",
  "abi-r1:cuda:navatala_cfd_pack_owner_vec3",
  "abi-r1:cuda:navatala_cfd_pack_owner_vec3",
  7,
  kAbiArgs_cuda_navatala_cfd_pack_owner_vec3
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pack_owner_vec3_mu[] = {
  { "ux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pack_owner_vec3_mu = {
  1,
  "navatala_cfd_pack_owner_vec3_mu",
  "cuda",
  "navatala_cfd_pack_owner_vec3_mu",
  "kernel:cuda:navatala_cfd_pack_owner_vec3_mu",
  "abi-r1:cuda:navatala_cfd_pack_owner_vec3_mu",
  "abi-r1:cuda:navatala_cfd_pack_owner_vec3_mu",
  8,
  kAbiArgs_cuda_navatala_cfd_pack_owner_vec3_mu
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scatter_mu_grad_ghost[] = {
  { "recvVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYX", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYY", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYZ", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZX", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZY", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZZ", 11, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scatter_mu_grad_ghost = {
  1,
  "navatala_cfd_scatter_mu_grad_ghost",
  "cuda",
  "navatala_cfd_scatter_mu_grad_ghost",
  "kernel:cuda:navatala_cfd_scatter_mu_grad_ghost",
  "abi-r1:cuda:navatala_cfd_scatter_mu_grad_ghost",
  "abi-r1:cuda:navatala_cfd_scatter_mu_grad_ghost",
  12,
  kAbiArgs_cuda_navatala_cfd_scatter_mu_grad_ghost
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scatter_rho_vec3_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcX", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcY", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcZ", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcMask", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scatter_rho_vec3_and_mask = {
  1,
  "navatala_cfd_scatter_rho_vec3_and_mask",
  "cuda",
  "navatala_cfd_scatter_rho_vec3_and_mask",
  "kernel:cuda:navatala_cfd_scatter_rho_vec3_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_rho_vec3_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_rho_vec3_and_mask",
  9,
  kAbiArgs_cuda_navatala_cfd_scatter_rho_vec3_and_mask
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scatter_vec3_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scatter_vec3_and_mask = {
  1,
  "navatala_cfd_scatter_vec3_and_mask",
  "cuda",
  "navatala_cfd_scatter_vec3_and_mask",
  "kernel:cuda:navatala_cfd_scatter_vec3_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_vec3_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_vec3_and_mask",
  7,
  kAbiArgs_cuda_navatala_cfd_scatter_vec3_and_mask
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scatter_vec3_mu_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBc", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scatter_vec3_mu_and_mask = {
  1,
  "navatala_cfd_scatter_vec3_mu_and_mask",
  "cuda",
  "navatala_cfd_scatter_vec3_mu_and_mask",
  "kernel:cuda:navatala_cfd_scatter_vec3_mu_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_vec3_mu_and_mask",
  "abi-r1:cuda:navatala_cfd_scatter_vec3_mu_and_mask",
  8,
  kAbiArgs_cuda_navatala_cfd_scatter_vec3_mu_and_mask
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_add_vol_vector[] = {
  { "ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ay", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "az", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "by_", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bz", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_add_vol_vector = {
  1,
  "navatala_cfd_add_vol_vector",
  "cuda",
  "navatala_cfd_add_vol_vector",
  "kernel:cuda:navatala_cfd_add_vol_vector",
  "abi-r1:cuda:navatala_cfd_add_vol_vector",
  "abi-r1:cuda:navatala_cfd_add_vol_vector",
  10,
  kAbiArgs_cuda_navatala_cfd_add_vol_vector
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_div_rho_phi_u_explicit[] = {
  { "phi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcX", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcY", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcZ", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcMask", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outX", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_div_rho_phi_u_explicit = {
  1,
  "navatala_cfd_div_rho_phi_u_explicit",
  "cuda",
  "navatala_cfd_div_rho_phi_u_explicit",
  "kernel:cuda:navatala_cfd_div_rho_phi_u_explicit",
  "abi-r1:cuda:navatala_cfd_div_rho_phi_u_explicit",
  "abi-r1:cuda:navatala_cfd_div_rho_phi_u_explicit",
  22,
  kAbiArgs_cuda_navatala_cfd_div_rho_phi_u_explicit
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_hby_a_from_u_and_terms[] = {
  { "ux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lapx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lapy", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lapz", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divx", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divy", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divz", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "extrax", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "extray", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "extraz", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rau", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modeBuf", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outx", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outy", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outz", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_hby_a_from_u_and_terms = {
  1,
  "navatala_cfd_hby_a_from_u_and_terms",
  "cuda",
  "navatala_cfd_hby_a_from_u_and_terms",
  "kernel:cuda:navatala_cfd_hby_a_from_u_and_terms",
  "abi-r1:cuda:navatala_cfd_hby_a_from_u_and_terms",
  "abi-r1:cuda:navatala_cfd_hby_a_from_u_and_terms",
  18,
  kAbiArgs_cuda_navatala_cfd_hby_a_from_u_and_terms
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_apply_sp_to_hby_a[] = {
  { "hx", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hy", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hz", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAU", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_apply_sp_to_hby_a = {
  1,
  "navatala_cfd_apply_sp_to_hby_a",
  "cuda",
  "navatala_cfd_apply_sp_to_hby_a",
  "kernel:cuda:navatala_cfd_apply_sp_to_hby_a",
  "abi-r1:cuda:navatala_cfd_apply_sp_to_hby_a",
  "abi-r1:cuda:navatala_cfd_apply_sp_to_hby_a",
  9,
  kAbiArgs_cuda_navatala_cfd_apply_sp_to_hby_a
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_laplacian_scalar_explicit[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMu", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMuMask", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcPhi", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcPhiMask", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outL", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_laplacian_scalar_explicit = {
  1,
  "navatala_cfd_laplacian_scalar_explicit",
  "cuda",
  "navatala_cfd_laplacian_scalar_explicit",
  "kernel:cuda:navatala_cfd_laplacian_scalar_explicit",
  "abi-r1:cuda:navatala_cfd_laplacian_scalar_explicit",
  "abi-r1:cuda:navatala_cfd_laplacian_scalar_explicit",
  17,
  kAbiArgs_cuda_navatala_cfd_laplacian_scalar_explicit
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_laplacian_vector_explicit[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBc", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "corrX", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "corrY", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "corrZ", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradRow0", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradRow1", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradRow2", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 19, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 20, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 21, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 22, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 23, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 24, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outX", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 27, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_laplacian_vector_explicit = {
  1,
  "navatala_cfd_laplacian_vector_explicit",
  "cuda",
  "navatala_cfd_laplacian_vector_explicit",
  "kernel:cuda:navatala_cfd_laplacian_vector_explicit",
  "abi-r1:cuda:navatala_cfd_laplacian_vector_explicit",
  "abi-r1:cuda:navatala_cfd_laplacian_vector_explicit",
  28,
  kAbiArgs_cuda_navatala_cfd_laplacian_vector_explicit
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_predictor_hby_a_from_source[] = {
  { "rAU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outHbyAX", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outHbyAY", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outHbyAZ", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_predictor_hby_a_from_source = {
  1,
  "navatala_cfd_predictor_hby_a_from_source",
  "cuda",
  "navatala_cfd_predictor_hby_a_from_source",
  "kernel:cuda:navatala_cfd_predictor_hby_a_from_source",
  "abi-r1:cuda:navatala_cfd_predictor_hby_a_from_source",
  "abi-r1:cuda:navatala_cfd_predictor_hby_a_from_source",
  8,
  kAbiArgs_cuda_navatala_cfd_predictor_hby_a_from_source
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_predictor_inverse_diag[] = {
  { "inputDiag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_predictor_inverse_diag = {
  1,
  "navatala_cfd_predictor_inverse_diag",
  "cuda",
  "navatala_cfd_predictor_inverse_diag",
  "kernel:cuda:navatala_cfd_predictor_inverse_diag",
  "abi-r1:cuda:navatala_cfd_predictor_inverse_diag",
  "abi-r1:cuda:navatala_cfd_predictor_inverse_diag",
  3,
  kAbiArgs_cuda_navatala_cfd_predictor_inverse_diag
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_predictor_phi_hby_a_fuse[] = {
  { "fluxHbyA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoRauFace", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ddtCorr", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhiHbyA", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_predictor_phi_hby_a_fuse = {
  1,
  "navatala_cfd_predictor_phi_hby_a_fuse",
  "cuda",
  "navatala_cfd_predictor_phi_hby_a_fuse",
  "kernel:cuda:navatala_cfd_predictor_phi_hby_a_fuse",
  "abi-r1:cuda:navatala_cfd_predictor_phi_hby_a_fuse",
  "abi-r1:cuda:navatala_cfd_predictor_phi_hby_a_fuse",
  5,
  kAbiArgs_cuda_navatala_cfd_predictor_phi_hby_a_fuse
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_axpy_cells[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_axpy_cells = {
  1,
  "navatala_cfd_axpy_cells",
  "cuda",
  "navatala_cfd_axpy_cells",
  "kernel:cuda:navatala_cfd_axpy_cells",
  "abi-r1:cuda:navatala_cfd_axpy_cells",
  "abi-r1:cuda:navatala_cfd_axpy_cells",
  4,
  kAbiArgs_cuda_navatala_cfd_axpy_cells
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_ddt_diagonal[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_ddt_diagonal = {
  1,
  "navatala_cfd_ddt_diagonal",
  "cuda",
  "navatala_cfd_ddt_diagonal",
  "kernel:cuda:navatala_cfd_ddt_diagonal",
  "abi-r1:cuda:navatala_cfd_ddt_diagonal",
  "abi-r1:cuda:navatala_cfd_ddt_diagonal",
  5,
  kAbiArgs_cuda_navatala_cfd_ddt_diagonal
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_div_upwind_coeffs[] = {
  { "weights", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "flux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outLower", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outUpper", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_div_upwind_coeffs = {
  1,
  "navatala_cfd_div_upwind_coeffs",
  "cuda",
  "navatala_cfd_div_upwind_coeffs",
  "kernel:cuda:navatala_cfd_div_upwind_coeffs",
  "abi-r1:cuda:navatala_cfd_div_upwind_coeffs",
  "abi-r1:cuda:navatala_cfd_div_upwind_coeffs",
  5,
  kAbiArgs_cuda_navatala_cfd_div_upwind_coeffs
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_fvm_sp[] = {
  { "sp", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_fvm_sp = {
  1,
  "navatala_cfd_fvm_sp",
  "cuda",
  "navatala_cfd_fvm_sp",
  "kernel:cuda:navatala_cfd_fvm_sp",
  "abi-r1:cuda:navatala_cfd_fvm_sp",
  "abi-r1:cuda:navatala_cfd_fvm_sp",
  4,
  kAbiArgs_cuda_navatala_cfd_fvm_sp
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_fvm_su_sp[] = {
  { "sp", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSource", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_fvm_su_sp = {
  1,
  "navatala_cfd_fvm_su_sp",
  "cuda",
  "navatala_cfd_fvm_su_sp",
  "kernel:cuda:navatala_cfd_fvm_su_sp",
  "abi-r1:cuda:navatala_cfd_fvm_su_sp",
  "abi-r1:cuda:navatala_cfd_fvm_su_sp",
  6,
  kAbiArgs_cuda_navatala_cfd_fvm_su_sp
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_laplacian_coeffs[] = {
  { "gamma", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outUpper", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outLower", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_laplacian_coeffs = {
  1,
  "navatala_cfd_laplacian_coeffs",
  "cuda",
  "navatala_cfd_laplacian_coeffs",
  "kernel:cuda:navatala_cfd_laplacian_coeffs",
  "abi-r1:cuda:navatala_cfd_laplacian_coeffs",
  "abi-r1:cuda:navatala_cfd_laplacian_coeffs",
  9,
  kAbiArgs_cuda_navatala_cfd_laplacian_coeffs
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_laplacian_coeffs_accum[] = {
  { "gamma", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outUpper", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outLower", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_laplacian_coeffs_accum = {
  1,
  "navatala_cfd_laplacian_coeffs_accum",
  "cuda",
  "navatala_cfd_laplacian_coeffs_accum",
  "kernel:cuda:navatala_cfd_laplacian_coeffs_accum",
  "abi-r1:cuda:navatala_cfd_laplacian_coeffs_accum",
  "abi-r1:cuda:navatala_cfd_laplacian_coeffs_accum",
  6,
  kAbiArgs_cuda_navatala_cfd_laplacian_coeffs_accum
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_linear_upwind_correction_source[] = {
  { "luFaceVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "flux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSource", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_linear_upwind_correction_source = {
  1,
  "navatala_cfd_linear_upwind_correction_source",
  "cuda",
  "navatala_cfd_linear_upwind_correction_source",
  "kernel:cuda:navatala_cfd_linear_upwind_correction_source",
  "abi-r1:cuda:navatala_cfd_linear_upwind_correction_source",
  "abi-r1:cuda:navatala_cfd_linear_upwind_correction_source",
  10,
  kAbiArgs_cuda_navatala_cfd_linear_upwind_correction_source
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_matrix_relax[] = {
  { "diagOld", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumOff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "source", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_matrix_relax = {
  1,
  "navatala_cfd_matrix_relax",
  "cuda",
  "navatala_cfd_matrix_relax",
  "kernel:cuda:navatala_cfd_matrix_relax",
  "abi-r1:cuda:navatala_cfd_matrix_relax",
  "abi-r1:cuda:navatala_cfd_matrix_relax",
  7,
  kAbiArgs_cuda_navatala_cfd_matrix_relax
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_neg_sum_diag[] = {
  { "upper", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outDiag", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_neg_sum_diag = {
  1,
  "navatala_cfd_neg_sum_diag",
  "cuda",
  "navatala_cfd_neg_sum_diag",
  "kernel:cuda:navatala_cfd_neg_sum_diag",
  "abi-r1:cuda:navatala_cfd_neg_sum_diag",
  "abi-r1:cuda:navatala_cfd_neg_sum_diag",
  7,
  kAbiArgs_cuda_navatala_cfd_neg_sum_diag
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pack_boundary_adjacent_cells[] = {
  { "cellField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indexMap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPacked", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pack_boundary_adjacent_cells = {
  1,
  "navatala_cfd_pack_boundary_adjacent_cells",
  "cuda",
  "navatala_cfd_pack_boundary_adjacent_cells",
  "kernel:cuda:navatala_cfd_pack_boundary_adjacent_cells",
  "abi-r1:cuda:navatala_cfd_pack_boundary_adjacent_cells",
  "abi-r1:cuda:navatala_cfd_pack_boundary_adjacent_cells",
  4,
  kAbiArgs_cuda_navatala_cfd_pack_boundary_adjacent_cells
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scalar_jacobi_clamp_min[] = {
  { "lowerBound", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scalar_jacobi_clamp_min = {
  1,
  "navatala_cfd_scalar_jacobi_clamp_min",
  "cuda",
  "navatala_cfd_scalar_jacobi_clamp_min",
  "kernel:cuda:navatala_cfd_scalar_jacobi_clamp_min",
  "abi-r1:cuda:navatala_cfd_scalar_jacobi_clamp_min",
  "abi-r1:cuda:navatala_cfd_scalar_jacobi_clamp_min",
  3,
  kAbiArgs_cuda_navatala_cfd_scalar_jacobi_clamp_min
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scalar_jacobi_update[] = {
  { "ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhs", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scalar_jacobi_update = {
  1,
  "navatala_cfd_scalar_jacobi_update",
  "cuda",
  "navatala_cfd_scalar_jacobi_update",
  "kernel:cuda:navatala_cfd_scalar_jacobi_update",
  "abi-r1:cuda:navatala_cfd_scalar_jacobi_update",
  "abi-r1:cuda:navatala_cfd_scalar_jacobi_update",
  6,
  kAbiArgs_cuda_navatala_cfd_scalar_jacobi_update
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scalar_ldu_mat_vec[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ax", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scalar_ldu_mat_vec = {
  1,
  "navatala_cfd_scalar_ldu_mat_vec",
  "cuda",
  "navatala_cfd_scalar_ldu_mat_vec",
  "kernel:cuda:navatala_cfd_scalar_ldu_mat_vec",
  "abi-r1:cuda:navatala_cfd_scalar_ldu_mat_vec",
  "abi-r1:cuda:navatala_cfd_scalar_ldu_mat_vec",
  11,
  kAbiArgs_cuda_navatala_cfd_scalar_ldu_mat_vec
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scalar_ldu_multi_color_gs_sweep[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhs", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOrder", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorRange", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 13, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scalar_ldu_multi_color_gs_sweep = {
  1,
  "navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "cuda",
  "navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "kernel:cuda:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "abi-r1:cuda:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "abi-r1:cuda:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  14,
  kAbiArgs_cuda_navatala_cfd_scalar_ldu_multi_color_gs_sweep
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_sum_mag_off_diag[] = {
  { "upper", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSumOff", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_sum_mag_off_diag = {
  1,
  "navatala_cfd_sum_mag_off_diag",
  "cuda",
  "navatala_cfd_sum_mag_off_diag",
  "kernel:cuda:navatala_cfd_sum_mag_off_diag",
  "abi-r1:cuda:navatala_cfd_sum_mag_off_diag",
  "abi-r1:cuda:navatala_cfd_sum_mag_off_diag",
  7,
  kAbiArgs_cuda_navatala_cfd_sum_mag_off_diag
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_axpy_in_place[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_axpy_in_place = {
  1,
  "navatala_cfd_gamg_axpy_in_place",
  "cuda",
  "navatala_cfd_gamg_axpy_in_place",
  "kernel:cuda:navatala_cfd_gamg_axpy_in_place",
  "abi-r1:cuda:navatala_cfd_gamg_axpy_in_place",
  "abi-r1:cuda:navatala_cfd_gamg_axpy_in_place",
  4,
  kAbiArgs_cuda_navatala_cfd_gamg_axpy_in_place
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_compute_scale_factor[] = {
  { "num", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "den", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sf", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_compute_scale_factor = {
  1,
  "navatala_cfd_gamg_compute_scale_factor",
  "cuda",
  "navatala_cfd_gamg_compute_scale_factor",
  "kernel:cuda:navatala_cfd_gamg_compute_scale_factor",
  "abi-r1:cuda:navatala_cfd_gamg_compute_scale_factor",
  "abi-r1:cuda:navatala_cfd_gamg_compute_scale_factor",
  3,
  kAbiArgs_cuda_navatala_cfd_gamg_compute_scale_factor
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_dot_finalize[] = {
  { "partials", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_dot_finalize = {
  1,
  "navatala_cfd_gamg_dot_finalize",
  "cuda",
  "navatala_cfd_gamg_dot_finalize",
  "kernel:cuda:navatala_cfd_gamg_dot_finalize",
  "abi-r1:cuda:navatala_cfd_gamg_dot_finalize",
  "abi-r1:cuda:navatala_cfd_gamg_dot_finalize",
  3,
  kAbiArgs_cuda_navatala_cfd_gamg_dot_finalize
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_negate_scalar[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_negate_scalar = {
  1,
  "navatala_cfd_gamg_negate_scalar",
  "cuda",
  "navatala_cfd_gamg_negate_scalar",
  "kernel:cuda:navatala_cfd_gamg_negate_scalar",
  "abi-r1:cuda:navatala_cfd_gamg_negate_scalar",
  "abi-r1:cuda:navatala_cfd_gamg_negate_scalar",
  2,
  kAbiArgs_cuda_navatala_cfd_gamg_negate_scalar
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_scale_correction[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Az", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfPtr", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "corr", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_scale_correction = {
  1,
  "navatala_cfd_gamg_scale_correction",
  "cuda",
  "navatala_cfd_gamg_scale_correction",
  "kernel:cuda:navatala_cfd_gamg_scale_correction",
  "abi-r1:cuda:navatala_cfd_gamg_scale_correction",
  "abi-r1:cuda:navatala_cfd_gamg_scale_correction",
  7,
  kAbiArgs_cuda_navatala_cfd_gamg_scale_correction
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_dic_build[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeU", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_dic_build = {
  1,
  "navatala_cfd_mg_dic_build",
  "cuda",
  "navatala_cfd_mg_dic_build",
  "kernel:cuda:navatala_cfd_mg_dic_build",
  "abi-r1:cuda:navatala_cfd_mg_dic_build",
  "abi-r1:cuda:navatala_cfd_mg_dic_build",
  6,
  kAbiArgs_cuda_navatala_cfd_mg_dic_build
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_dic_smooth[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeU", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_dic_smooth = {
  1,
  "navatala_cfd_mg_dic_smooth",
  "cuda",
  "navatala_cfd_mg_dic_smooth",
  "kernel:cuda:navatala_cfd_mg_dic_smooth",
  "abi-r1:cuda:navatala_cfd_mg_dic_smooth",
  "abi-r1:cuda:navatala_cfd_mg_dic_smooth",
  9,
  kAbiArgs_cuda_navatala_cfd_mg_dic_smooth
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_coarse_jacobi[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_coarse_jacobi = {
  1,
  "navatala_cfd_mg_coarse_jacobi",
  "cuda",
  "navatala_cfd_mg_coarse_jacobi",
  "kernel:cuda:navatala_cfd_mg_coarse_jacobi",
  "abi-r1:cuda:navatala_cfd_mg_coarse_jacobi",
  "abi-r1:cuda:navatala_cfd_mg_coarse_jacobi",
  4,
  kAbiArgs_cuda_navatala_cfd_mg_coarse_jacobi
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_coarse_matvec[] = {
  { "edgeU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_coarse_matvec = {
  1,
  "navatala_cfd_mg_coarse_matvec",
  "cuda",
  "navatala_cfd_mg_coarse_matvec",
  "kernel:cuda:navatala_cfd_mg_coarse_matvec",
  "abi-r1:cuda:navatala_cfd_mg_coarse_matvec",
  "abi-r1:cuda:navatala_cfd_mg_coarse_matvec",
  7,
  kAbiArgs_cuda_navatala_cfd_mg_coarse_matvec
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_coarse_rhs[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_coarse_rhs = {
  1,
  "navatala_cfd_mg_coarse_rhs",
  "cuda",
  "navatala_cfd_mg_coarse_rhs",
  "kernel:cuda:navatala_cfd_mg_coarse_rhs",
  "abi-r1:cuda:navatala_cfd_mg_coarse_rhs",
  "abi-r1:cuda:navatala_cfd_mg_coarse_rhs",
  4,
  kAbiArgs_cuda_navatala_cfd_mg_coarse_rhs
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_find_best_match[] = {
  { "csrOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csrFaceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csrSign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strength", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bestNeighbor", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_find_best_match = {
  1,
  "navatala_cfd_mg_find_best_match",
  "cuda",
  "navatala_cfd_mg_find_best_match",
  "kernel:cuda:navatala_cfd_mg_find_best_match",
  "abi-r1:cuda:navatala_cfd_mg_find_best_match",
  "abi-r1:cuda:navatala_cfd_mg_find_best_match",
  8,
  kAbiArgs_cuda_navatala_cfd_mg_find_best_match
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_pair_match[] = {
  { "bestNeighbor", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "aggMap", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_pair_match = {
  1,
  "navatala_cfd_mg_pair_match",
  "cuda",
  "navatala_cfd_mg_pair_match",
  "kernel:cuda:navatala_cfd_mg_pair_match",
  "abi-r1:cuda:navatala_cfd_mg_pair_match",
  "abi-r1:cuda:navatala_cfd_mg_pair_match",
  3,
  kAbiArgs_cuda_navatala_cfd_mg_pair_match
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_prolong_add[] = {
  { "xCoarse", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongRowOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "zFine", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_prolong_add = {
  1,
  "navatala_cfd_mg_prolong_add",
  "cuda",
  "navatala_cfd_mg_prolong_add",
  "kernel:cuda:navatala_cfd_mg_prolong_add",
  "abi-r1:cuda:navatala_cfd_mg_prolong_add",
  "abi-r1:cuda:navatala_cfd_mg_prolong_add",
  6,
  kAbiArgs_cuda_navatala_cfd_mg_prolong_add
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_renumber_aggregates[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counter", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_renumber_aggregates = {
  1,
  "navatala_cfd_mg_renumber_aggregates",
  "cuda",
  "navatala_cfd_mg_renumber_aggregates",
  "kernel:cuda:navatala_cfd_mg_renumber_aggregates",
  "abi-r1:cuda:navatala_cfd_mg_renumber_aggregates",
  "abi-r1:cuda:navatala_cfd_mg_renumber_aggregates",
  3,
  kAbiArgs_cuda_navatala_cfd_mg_renumber_aggregates
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_renumber_propagate[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_renumber_propagate = {
  1,
  "navatala_cfd_mg_renumber_propagate",
  "cuda",
  "navatala_cfd_mg_renumber_propagate",
  "kernel:cuda:navatala_cfd_mg_renumber_propagate",
  "abi-r1:cuda:navatala_cfd_mg_renumber_propagate",
  "abi-r1:cuda:navatala_cfd_mg_renumber_propagate",
  2,
  kAbiArgs_cuda_navatala_cfd_mg_renumber_propagate
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_resolve_chains[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_resolve_chains = {
  1,
  "navatala_cfd_mg_resolve_chains",
  "cuda",
  "navatala_cfd_mg_resolve_chains",
  "kernel:cuda:navatala_cfd_mg_resolve_chains",
  "abi-r1:cuda:navatala_cfd_mg_resolve_chains",
  "abi-r1:cuda:navatala_cfd_mg_resolve_chains",
  2,
  kAbiArgs_cuda_navatala_cfd_mg_resolve_chains
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_restrict_sum[] = {
  { "rFine", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongRowOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rCoarse", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_restrict_sum = {
  1,
  "navatala_cfd_mg_restrict_sum",
  "cuda",
  "navatala_cfd_mg_restrict_sum",
  "kernel:cuda:navatala_cfd_mg_restrict_sum",
  "abi-r1:cuda:navatala_cfd_mg_restrict_sum",
  "abi-r1:cuda:navatala_cfd_mg_restrict_sum",
  6,
  kAbiArgs_cuda_navatala_cfd_mg_restrict_sum
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_strength_of_connection[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strength", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_strength_of_connection = {
  1,
  "navatala_cfd_mg_strength_of_connection",
  "cuda",
  "navatala_cfd_mg_strength_of_connection",
  "kernel:cuda:navatala_cfd_mg_strength_of_connection",
  "abi-r1:cuda:navatala_cfd_mg_strength_of_connection",
  "abi-r1:cuda:navatala_cfd_mg_strength_of_connection",
  6,
  kAbiArgs_cuda_navatala_cfd_mg_strength_of_connection
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mg_zero_coarse[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mg_zero_coarse = {
  1,
  "navatala_cfd_mg_zero_coarse",
  "cuda",
  "navatala_cfd_mg_zero_coarse",
  "kernel:cuda:navatala_cfd_mg_zero_coarse",
  "abi-r1:cuda:navatala_cfd_mg_zero_coarse",
  "abi-r1:cuda:navatala_cfd_mg_zero_coarse",
  2,
  kAbiArgs_cuda_navatala_cfd_mg_zero_coarse
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_phi_correct[] = {
  { "phiIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "paramsF", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "phiOut", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_phi_correct = {
  1,
  "navatala_cfd_phi_correct",
  "cuda",
  "navatala_cfd_phi_correct",
  "kernel:cuda:navatala_cfd_phi_correct",
  "abi-r1:cuda:navatala_cfd_phi_correct",
  "abi-r1:cuda:navatala_cfd_phi_correct",
  15,
  kAbiArgs_cuda_navatala_cfd_phi_correct
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_phi_correct_boundary[] = {
  { "phiIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "paramsF", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "phiOut", 13, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_phi_correct_boundary = {
  1,
  "navatala_cfd_phi_correct_boundary",
  "cuda",
  "navatala_cfd_phi_correct_boundary",
  "kernel:cuda:navatala_cfd_phi_correct_boundary",
  "abi-r1:cuda:navatala_cfd_phi_correct_boundary",
  "abi-r1:cuda:navatala_cfd_phi_correct_boundary",
  14,
  kAbiArgs_cuda_navatala_cfd_phi_correct_boundary
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_phi_correct_internal[] = {
  { "phiIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "paramsF", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "phiOut", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_phi_correct_internal = {
  1,
  "navatala_cfd_phi_correct_internal",
  "cuda",
  "navatala_cfd_phi_correct_internal",
  "kernel:cuda:navatala_cfd_phi_correct_internal",
  "abi-r1:cuda:navatala_cfd_phi_correct_internal",
  "abi-r1:cuda:navatala_cfd_phi_correct_internal",
  11,
  kAbiArgs_cuda_navatala_cfd_phi_correct_internal
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_axpy_in_place[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_axpy_in_place = {
  1,
  "navatala_cfd_axpy_in_place",
  "cuda",
  "navatala_cfd_axpy_in_place",
  "kernel:cuda:navatala_cfd_axpy_in_place",
  "abi-r1:cuda:navatala_cfd_axpy_in_place",
  "abi-r1:cuda:navatala_cfd_axpy_in_place",
  4,
  kAbiArgs_cuda_navatala_cfd_axpy_in_place
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_bc_dirichlet_face_flux[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_bc_dirichlet_face_flux = {
  1,
  "navatala_cfd_bc_dirichlet_face_flux",
  "cuda",
  "navatala_cfd_bc_dirichlet_face_flux",
  "kernel:cuda:navatala_cfd_bc_dirichlet_face_flux",
  "abi-r1:cuda:navatala_cfd_bc_dirichlet_face_flux",
  "abi-r1:cuda:navatala_cfd_bc_dirichlet_face_flux",
  5,
  kAbiArgs_cuda_navatala_cfd_bc_dirichlet_face_flux
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_bc_sn_grad_face_flux[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_bc_sn_grad_face_flux = {
  1,
  "navatala_cfd_bc_sn_grad_face_flux",
  "cuda",
  "navatala_cfd_bc_sn_grad_face_flux",
  "kernel:cuda:navatala_cfd_bc_sn_grad_face_flux",
  "abi-r1:cuda:navatala_cfd_bc_sn_grad_face_flux",
  "abi-r1:cuda:navatala_cfd_bc_sn_grad_face_flux",
  6,
  kAbiArgs_cuda_navatala_cfd_bc_sn_grad_face_flux
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_coeff_to_cf_in_place[] = {
  { "cf", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_coeff_to_cf_in_place = {
  1,
  "navatala_cfd_coeff_to_cf_in_place",
  "cuda",
  "navatala_cfd_coeff_to_cf_in_place",
  "kernel:cuda:navatala_cfd_coeff_to_cf_in_place",
  "abi-r1:cuda:navatala_cfd_coeff_to_cf_in_place",
  "abi-r1:cuda:navatala_cfd_coeff_to_cf_in_place",
  4,
  kAbiArgs_cuda_navatala_cfd_coeff_to_cf_in_place
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_diag_from_cf[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_diag_from_cf = {
  1,
  "navatala_cfd_diag_from_cf",
  "cuda",
  "navatala_cfd_diag_from_cf",
  "kernel:cuda:navatala_cfd_diag_from_cf",
  "abi-r1:cuda:navatala_cfd_diag_from_cf",
  "abi-r1:cuda:navatala_cfd_diag_from_cf",
  6,
  kAbiArgs_cuda_navatala_cfd_diag_from_cf
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_dic_apply[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "z", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_dic_apply = {
  1,
  "navatala_cfd_dic_apply",
  "cuda",
  "navatala_cfd_dic_apply",
  "kernel:cuda:navatala_cfd_dic_apply",
  "abi-r1:cuda:navatala_cfd_dic_apply",
  "abi-r1:cuda:navatala_cfd_dic_apply",
  7,
  kAbiArgs_cuda_navatala_cfd_dic_apply
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_dic_build_reciprocal_d[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "rD", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_dic_build_reciprocal_d = {
  1,
  "navatala_cfd_dic_build_reciprocal_d",
  "cuda",
  "navatala_cfd_dic_build_reciprocal_d",
  "kernel:cuda:navatala_cfd_dic_build_reciprocal_d",
  "abi-r1:cuda:navatala_cfd_dic_build_reciprocal_d",
  "abi-r1:cuda:navatala_cfd_dic_build_reciprocal_d",
  6,
  kAbiArgs_cuda_navatala_cfd_dic_build_reciprocal_d
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_dot_partials[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_dot_partials = {
  1,
  "navatala_cfd_dot_partials",
  "cuda",
  "navatala_cfd_dot_partials",
  "kernel:cuda:navatala_cfd_dot_partials",
  "abi-r1:cuda:navatala_cfd_dot_partials",
  "abi-r1:cuda:navatala_cfd_dot_partials",
  4,
  kAbiArgs_cuda_navatala_cfd_dot_partials
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_face_flux[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_face_flux = {
  1,
  "navatala_cfd_face_flux",
  "cuda",
  "navatala_cfd_face_flux",
  "kernel:cuda:navatala_cfd_face_flux",
  "abi-r1:cuda:navatala_cfd_face_flux",
  "abi-r1:cuda:navatala_cfd_face_flux",
  8,
  kAbiArgs_cuda_navatala_cfd_face_flux
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_inv_diag[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "z", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_inv_diag = {
  1,
  "navatala_cfd_inv_diag",
  "cuda",
  "navatala_cfd_inv_diag",
  "kernel:cuda:navatala_cfd_inv_diag",
  "abi-r1:cuda:navatala_cfd_inv_diag",
  "abi-r1:cuda:navatala_cfd_inv_diag",
  4,
  kAbiArgs_cuda_navatala_cfd_inv_diag
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_laplacian_from_face_flux[] = {
  { "faceFlux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outLap", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_laplacian_from_face_flux = {
  1,
  "navatala_cfd_laplacian_from_face_flux",
  "cuda",
  "navatala_cfd_laplacian_from_face_flux",
  "kernel:cuda:navatala_cfd_laplacian_from_face_flux",
  "abi-r1:cuda:navatala_cfd_laplacian_from_face_flux",
  "abi-r1:cuda:navatala_cfd_laplacian_from_face_flux",
  7,
  kAbiArgs_cuda_navatala_cfd_laplacian_from_face_flux
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mul_by_vol_in_place[] = {
  { "b", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mul_by_vol_in_place = {
  1,
  "navatala_cfd_mul_by_vol_in_place",
  "cuda",
  "navatala_cfd_mul_by_vol_in_place",
  "kernel:cuda:navatala_cfd_mul_by_vol_in_place",
  "abi-r1:cuda:navatala_cfd_mul_by_vol_in_place",
  "abi-r1:cuda:navatala_cfd_mul_by_vol_in_place",
  3,
  kAbiArgs_cuda_navatala_cfd_mul_by_vol_in_place
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_ref_add_ax[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refCell", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_ref_add_ax = {
  1,
  "navatala_cfd_ref_add_ax",
  "cuda",
  "navatala_cfd_ref_add_ax",
  "kernel:cuda:navatala_cfd_ref_add_ax",
  "abi-r1:cuda:navatala_cfd_ref_add_ax",
  "abi-r1:cuda:navatala_cfd_ref_add_ax",
  4,
  kAbiArgs_cuda_navatala_cfd_ref_add_ax
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_ref_add_b[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refCell", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_ref_add_b = {
  1,
  "navatala_cfd_ref_add_b",
  "cuda",
  "navatala_cfd_ref_add_b",
  "kernel:cuda:navatala_cfd_ref_add_b",
  "abi-r1:cuda:navatala_cfd_ref_add_b",
  "abi-r1:cuda:navatala_cfd_ref_add_b",
  4,
  kAbiArgs_cuda_navatala_cfd_ref_add_b
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_rhs_sub[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_rhs_sub = {
  1,
  "navatala_cfd_rhs_sub",
  "cuda",
  "navatala_cfd_rhs_sub",
  "kernel:cuda:navatala_cfd_rhs_sub",
  "abi-r1:cuda:navatala_cfd_rhs_sub",
  "abi-r1:cuda:navatala_cfd_rhs_sub",
  4,
  kAbiArgs_cuda_navatala_cfd_rhs_sub
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_shift_in_place[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_shift_in_place = {
  1,
  "navatala_cfd_shift_in_place",
  "cuda",
  "navatala_cfd_shift_in_place",
  "kernel:cuda:navatala_cfd_shift_in_place",
  "abi-r1:cuda:navatala_cfd_shift_in_place",
  "abi-r1:cuda:navatala_cfd_shift_in_place",
  3,
  kAbiArgs_cuda_navatala_cfd_shift_in_place
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_sum_abs_partials[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPartials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_sum_abs_partials = {
  1,
  "navatala_cfd_sum_abs_partials",
  "cuda",
  "navatala_cfd_sum_abs_partials",
  "kernel:cuda:navatala_cfd_sum_abs_partials",
  "abi-r1:cuda:navatala_cfd_sum_abs_partials",
  "abi-r1:cuda:navatala_cfd_sum_abs_partials",
  3,
  kAbiArgs_cuda_navatala_cfd_sum_abs_partials
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_upper_from_cf[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_upper_from_cf = {
  1,
  "navatala_cfd_upper_from_cf",
  "cuda",
  "navatala_cfd_upper_from_cf",
  "kernel:cuda:navatala_cfd_upper_from_cf",
  "abi-r1:cuda:navatala_cfd_upper_from_cf",
  "abi-r1:cuda:navatala_cfd_upper_from_cf",
  3,
  kAbiArgs_cuda_navatala_cfd_upper_from_cf
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_xpay_in_place[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pOut", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_xpay_in_place = {
  1,
  "navatala_cfd_xpay_in_place",
  "cuda",
  "navatala_cfd_xpay_in_place",
  "kernel:cuda:navatala_cfd_xpay_in_place",
  "abi-r1:cuda:navatala_cfd_xpay_in_place",
  "abi-r1:cuda:navatala_cfd_xpay_in_place",
  5,
  kAbiArgs_cuda_navatala_cfd_xpay_in_place
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_dot_finalize_partials[] = {
  { "partials", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_dot_finalize_partials = {
  1,
  "navatala_cfd_dot_finalize_partials",
  "cuda",
  "navatala_cfd_dot_finalize_partials",
  "kernel:cuda:navatala_cfd_dot_finalize_partials",
  "abi-r1:cuda:navatala_cfd_dot_finalize_partials",
  "abi-r1:cuda:navatala_cfd_dot_finalize_partials",
  3,
  kAbiArgs_cuda_navatala_cfd_dot_finalize_partials
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_negate_scalar[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_negate_scalar = {
  1,
  "navatala_cfd_negate_scalar",
  "cuda",
  "navatala_cfd_negate_scalar",
  "kernel:cuda:navatala_cfd_negate_scalar",
  "abi-r1:cuda:navatala_cfd_negate_scalar",
  "abi-r1:cuda:navatala_cfd_negate_scalar",
  2,
  kAbiArgs_cuda_navatala_cfd_negate_scalar
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pcg_compute_alpha[] = {
  { "rz", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "pAp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pcg_compute_alpha = {
  1,
  "navatala_cfd_pcg_compute_alpha",
  "cuda",
  "navatala_cfd_pcg_compute_alpha",
  "kernel:cuda:navatala_cfd_pcg_compute_alpha",
  "abi-r1:cuda:navatala_cfd_pcg_compute_alpha",
  "abi-r1:cuda:navatala_cfd_pcg_compute_alpha",
  3,
  kAbiArgs_cuda_navatala_cfd_pcg_compute_alpha
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pcg_compute_beta[] = {
  { "rzNew", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "rzOld", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pcg_compute_beta = {
  1,
  "navatala_cfd_pcg_compute_beta",
  "cuda",
  "navatala_cfd_pcg_compute_beta",
  "kernel:cuda:navatala_cfd_pcg_compute_beta",
  "abi-r1:cuda:navatala_cfd_pcg_compute_beta",
  "abi-r1:cuda:navatala_cfd_pcg_compute_beta",
  3,
  kAbiArgs_cuda_navatala_cfd_pcg_compute_beta
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_exact_mat_vec[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_exact_mat_vec = {
  1,
  "navatala_cfd_exact_mat_vec",
  "cuda",
  "navatala_cfd_exact_mat_vec",
  "kernel:cuda:navatala_cfd_exact_mat_vec",
  "abi-r1:cuda:navatala_cfd_exact_mat_vec",
  "abi-r1:cuda:navatala_cfd_exact_mat_vec",
  11,
  kAbiArgs_cuda_navatala_cfd_exact_mat_vec
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_exact_mat_vec_f64[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_exact_mat_vec_f64 = {
  1,
  "navatala_cfd_exact_mat_vec_f64",
  "cuda",
  "navatala_cfd_exact_mat_vec_f64",
  "kernel:cuda:navatala_cfd_exact_mat_vec_f64",
  "abi-r1:cuda:navatala_cfd_exact_mat_vec_f64",
  "abi-r1:cuda:navatala_cfd_exact_mat_vec_f64",
  11,
  kAbiArgs_cuda_navatala_cfd_exact_mat_vec_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_face_flux_boundary[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceFlux", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_face_flux_boundary = {
  1,
  "navatala_cfd_face_flux_boundary",
  "cuda",
  "navatala_cfd_face_flux_boundary",
  "kernel:cuda:navatala_cfd_face_flux_boundary",
  "abi-r1:cuda:navatala_cfd_face_flux_boundary",
  "abi-r1:cuda:navatala_cfd_face_flux_boundary",
  7,
  kAbiArgs_cuda_navatala_cfd_face_flux_boundary
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_face_flux_internal[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceFlux", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_face_flux_internal = {
  1,
  "navatala_cfd_face_flux_internal",
  "cuda",
  "navatala_cfd_face_flux_internal",
  "kernel:cuda:navatala_cfd_face_flux_internal",
  "abi-r1:cuda:navatala_cfd_face_flux_internal",
  "abi-r1:cuda:navatala_cfd_face_flux_internal",
  6,
  kAbiArgs_cuda_navatala_cfd_face_flux_internal
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pressure_face_flux_correction_corrected[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dCorrX", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dCorrY", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dCorrZ", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outFaceFlux", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pressure_face_flux_correction_corrected = {
  1,
  "navatala_cfd_pressure_face_flux_correction_corrected",
  "cuda",
  "navatala_cfd_pressure_face_flux_correction_corrected",
  "kernel:cuda:navatala_cfd_pressure_face_flux_correction_corrected",
  "abi-r1:cuda:navatala_cfd_pressure_face_flux_correction_corrected",
  "abi-r1:cuda:navatala_cfd_pressure_face_flux_correction_corrected",
  13,
  kAbiArgs_cuda_navatala_cfd_pressure_face_flux_correction_corrected
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_pressure_sn_grad[] = {
  { "phiHbyA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerAll", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcX", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcY", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcZ", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcMask", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "snGradMask", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSnGrad", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_pressure_sn_grad = {
  1,
  "navatala_cfd_pressure_sn_grad",
  "cuda",
  "navatala_cfd_pressure_sn_grad",
  "kernel:cuda:navatala_cfd_pressure_sn_grad",
  "abi-r1:cuda:navatala_cfd_pressure_sn_grad",
  "abi-r1:cuda:navatala_cfd_pressure_sn_grad",
  17,
  kAbiArgs_cuda_navatala_cfd_pressure_sn_grad
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_u_correct[] = {
  { "pCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAU", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_u_correct = {
  1,
  "navatala_cfd_u_correct",
  "cuda",
  "navatala_cfd_u_correct",
  "kernel:cuda:navatala_cfd_u_correct",
  "abi-r1:cuda:navatala_cfd_u_correct",
  "abi-r1:cuda:navatala_cfd_u_correct",
  21,
  kAbiArgs_cuda_navatala_cfd_u_correct
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_u_correct_reconstruct[] = {
  { "pCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAU", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hbx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hby", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hbz", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phig", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 19, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 20, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 21, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 22, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 23, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 24, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 27, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_u_correct_reconstruct = {
  1,
  "navatala_cfd_u_correct_reconstruct",
  "cuda",
  "navatala_cfd_u_correct_reconstruct",
  "kernel:cuda:navatala_cfd_u_correct_reconstruct",
  "abi-r1:cuda:navatala_cfd_u_correct_reconstruct",
  "abi-r1:cuda:navatala_cfd_u_correct_reconstruct",
  28,
  kAbiArgs_cuda_navatala_cfd_u_correct_reconstruct
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_grad_alpha_cell[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_grad_alpha_cell = {
  1,
  "navatala_cfd_grad_alpha_cell",
  "cuda",
  "navatala_cfd_grad_alpha_cell",
  "kernel:cuda:navatala_cfd_grad_alpha_cell",
  "abi-r1:cuda:navatala_cfd_grad_alpha_cell",
  "abi-r1:cuda:navatala_cfd_grad_alpha_cell",
  12,
  kAbiArgs_cuda_navatala_cfd_grad_alpha_cell
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_grad_alpha_cell_f64[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_grad_alpha_cell_f64 = {
  1,
  "navatala_cfd_grad_alpha_cell_f64",
  "cuda",
  "navatala_cfd_grad_alpha_cell_f64",
  "kernel:cuda:navatala_cfd_grad_alpha_cell_f64",
  "abi-r1:cuda:navatala_cfd_grad_alpha_cell_f64",
  "abi-r1:cuda:navatala_cfd_grad_alpha_cell_f64",
  12,
  kAbiArgs_cuda_navatala_cfd_grad_alpha_cell_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_n_hatf_all[] = {
  { "gx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "contactGrad", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nHatf", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_n_hatf_all = {
  1,
  "navatala_cfd_n_hatf_all",
  "cuda",
  "navatala_cfd_n_hatf_all",
  "kernel:cuda:navatala_cfd_n_hatf_all",
  "abi-r1:cuda:navatala_cfd_n_hatf_all",
  "abi-r1:cuda:navatala_cfd_n_hatf_all",
  15,
  kAbiArgs_cuda_navatala_cfd_n_hatf_all
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_n_hatf_all_f64[] = {
  { "gx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "contactGrad", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nHatf", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_n_hatf_all_f64 = {
  1,
  "navatala_cfd_n_hatf_all_f64",
  "cuda",
  "navatala_cfd_n_hatf_all_f64",
  "kernel:cuda:navatala_cfd_n_hatf_all_f64",
  "abi-r1:cuda:navatala_cfd_n_hatf_all_f64",
  "abi-r1:cuda:navatala_cfd_n_hatf_all_f64",
  15,
  kAbiArgs_cuda_navatala_cfd_n_hatf_all_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_stf_all_faces[] = {
  { "divN", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contactGrad", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "stf", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_stf_all_faces = {
  1,
  "navatala_cfd_stf_all_faces",
  "cuda",
  "navatala_cfd_stf_all_faces",
  "kernel:cuda:navatala_cfd_stf_all_faces",
  "abi-r1:cuda:navatala_cfd_stf_all_faces",
  "abi-r1:cuda:navatala_cfd_stf_all_faces",
  12,
  kAbiArgs_cuda_navatala_cfd_stf_all_faces
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_stf_all_faces_f64[] = {
  { "divN", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contactGrad", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "stf", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_stf_all_faces_f64 = {
  1,
  "navatala_cfd_stf_all_faces_f64",
  "cuda",
  "navatala_cfd_stf_all_faces_f64",
  "kernel:cuda:navatala_cfd_stf_all_faces_f64",
  "abi-r1:cuda:navatala_cfd_stf_all_faces_f64",
  "abi-r1:cuda:navatala_cfd_stf_all_faces_f64",
  12,
  kAbiArgs_cuda_navatala_cfd_stf_all_faces_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_k_omega_s_s_t_blending[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nu", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKy", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKz", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOmegax", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOmegay", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOmegaz", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outF1", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outF23", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outCDkOmega", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_k_omega_s_s_t_blending = {
  1,
  "navatala_cfd_k_omega_s_s_t_blending",
  "cuda",
  "navatala_cfd_k_omega_s_s_t_blending",
  "kernel:cuda:navatala_cfd_k_omega_s_s_t_blending",
  "abi-r1:cuda:navatala_cfd_k_omega_s_s_t_blending",
  "abi-r1:cuda:navatala_cfd_k_omega_s_s_t_blending",
  15,
  kAbiArgs_cuda_navatala_cfd_k_omega_s_s_t_blending
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_k_omega_s_s_t_nut_compute[] = {
  { "kVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "f23Vals", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s2Vals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outNut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_k_omega_s_s_t_nut_compute = {
  1,
  "navatala_cfd_k_omega_s_s_t_nut_compute",
  "cuda",
  "navatala_cfd_k_omega_s_s_t_nut_compute",
  "kernel:cuda:navatala_cfd_k_omega_s_s_t_nut_compute",
  "abi-r1:cuda:navatala_cfd_k_omega_s_s_t_nut_compute",
  "abi-r1:cuda:navatala_cfd_k_omega_s_s_t_nut_compute",
  7,
  kAbiArgs_cuda_navatala_cfd_k_omega_s_s_t_nut_compute
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_k_omega_s_s_t_sources[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nu", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rho", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "F1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "F23", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "CDkOmega", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "S2", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divU", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outOmegaSp", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outOmegaSuSp1", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outOmegaSuSp2", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outOmegaExpl", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outKSp", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outKSuSp", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outKExpl", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDomegaEff", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDkEff", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_k_omega_s_s_t_sources = {
  1,
  "navatala_cfd_k_omega_s_s_t_sources",
  "cuda",
  "navatala_cfd_k_omega_s_s_t_sources",
  "kernel:cuda:navatala_cfd_k_omega_s_s_t_sources",
  "abi-r1:cuda:navatala_cfd_k_omega_s_s_t_sources",
  "abi-r1:cuda:navatala_cfd_k_omega_s_s_t_sources",
  21,
  kAbiArgs_cuda_navatala_cfd_k_omega_s_s_t_sources
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mu_eff_from_rho_nu_eff[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nuEff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outMuEff", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mu_eff_from_rho_nu_eff = {
  1,
  "navatala_cfd_mu_eff_from_rho_nu_eff",
  "cuda",
  "navatala_cfd_mu_eff_from_rho_nu_eff",
  "kernel:cuda:navatala_cfd_mu_eff_from_rho_nu_eff",
  "abi-r1:cuda:navatala_cfd_mu_eff_from_rho_nu_eff",
  "abi-r1:cuda:navatala_cfd_mu_eff_from_rho_nu_eff",
  4,
  kAbiArgs_cuda_navatala_cfd_mu_eff_from_rho_nu_eff
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nuMix", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outMu", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix = {
  1,
  "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "cuda",
  "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "kernel:cuda:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "abi-r1:cuda:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "abi-r1:cuda:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  5,
  kAbiArgs_cuda_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_turbulence_add_explicit_source_device[] = {
  { "explicitSrc", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outVals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_turbulence_add_explicit_source_device = {
  1,
  "navatala_cfd_turbulence_add_explicit_source_device",
  "cuda",
  "navatala_cfd_turbulence_add_explicit_source_device",
  "kernel:cuda:navatala_cfd_turbulence_add_explicit_source_device",
  "abi-r1:cuda:navatala_cfd_turbulence_add_explicit_source_device",
  "abi-r1:cuda:navatala_cfd_turbulence_add_explicit_source_device",
  4,
  kAbiArgs_cuda_navatala_cfd_turbulence_add_explicit_source_device
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_turbulence_dirichlet_cell_source[] = {
  { "constrainedMask", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedValue", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "source", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_turbulence_dirichlet_cell_source = {
  1,
  "navatala_cfd_turbulence_dirichlet_cell_source",
  "cuda",
  "navatala_cfd_turbulence_dirichlet_cell_source",
  "kernel:cuda:navatala_cfd_turbulence_dirichlet_cell_source",
  "abi-r1:cuda:navatala_cfd_turbulence_dirichlet_cell_source",
  "abi-r1:cuda:navatala_cfd_turbulence_dirichlet_cell_source",
  5,
  kAbiArgs_cuda_navatala_cfd_turbulence_dirichlet_cell_source
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_turbulence_dirichlet_face_elimination[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedValue", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIntFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "upper", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "source", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_turbulence_dirichlet_face_elimination = {
  1,
  "navatala_cfd_turbulence_dirichlet_face_elimination",
  "cuda",
  "navatala_cfd_turbulence_dirichlet_face_elimination",
  "kernel:cuda:navatala_cfd_turbulence_dirichlet_face_elimination",
  "abi-r1:cuda:navatala_cfd_turbulence_dirichlet_face_elimination",
  "abi-r1:cuda:navatala_cfd_turbulence_dirichlet_face_elimination",
  8,
  kAbiArgs_cuda_navatala_cfd_turbulence_dirichlet_face_elimination
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_two_mag_sqr_symm_grad_u[] = {
  { "ufX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ufY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ufZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outS2", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_two_mag_sqr_symm_grad_u = {
  1,
  "navatala_cfd_two_mag_sqr_symm_grad_u",
  "cuda",
  "navatala_cfd_two_mag_sqr_symm_grad_u",
  "kernel:cuda:navatala_cfd_two_mag_sqr_symm_grad_u",
  "abi-r1:cuda:navatala_cfd_two_mag_sqr_symm_grad_u",
  "abi-r1:cuda:navatala_cfd_two_mag_sqr_symm_grad_u",
  12,
  kAbiArgs_cuda_navatala_cfd_two_mag_sqr_symm_grad_u
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_attention_row_softmax[] = {
  { "scores", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_attention_row_softmax = {
  1,
  "navatala_cfd_attention_row_softmax",
  "cuda",
  "navatala_cfd_attention_row_softmax",
  "kernel:cuda:navatala_cfd_attention_row_softmax",
  "abi-r1:cuda:navatala_cfd_attention_row_softmax",
  "abi-r1:cuda:navatala_cfd_attention_row_softmax",
  4,
  kAbiArgs_cuda_navatala_cfd_attention_row_softmax
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_attention_value_projection[] = {
  { "weights", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_attention_value_projection = {
  1,
  "navatala_cfd_attention_value_projection",
  "cuda",
  "navatala_cfd_attention_value_projection",
  "kernel:cuda:navatala_cfd_attention_value_projection",
  "abi-r1:cuda:navatala_cfd_attention_value_projection",
  "abi-r1:cuda:navatala_cfd_attention_value_projection",
  6,
  kAbiArgs_cuda_navatala_cfd_attention_value_projection
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gather_mean[] = {
  { "features", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighborCounts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numNodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "featureDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gather_mean = {
  1,
  "navatala_cfd_gather_mean",
  "cuda",
  "navatala_cfd_gather_mean",
  "kernel:cuda:navatala_cfd_gather_mean",
  "abi-r1:cuda:navatala_cfd_gather_mean",
  "abi-r1:cuda:navatala_cfd_gather_mean",
  6,
  kAbiArgs_cuda_navatala_cfd_gather_mean
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_layer_norm_forward[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_layer_norm_forward = {
  1,
  "navatala_cfd_layer_norm_forward",
  "cuda",
  "navatala_cfd_layer_norm_forward",
  "kernel:cuda:navatala_cfd_layer_norm_forward",
  "abi-r1:cuda:navatala_cfd_layer_norm_forward",
  "abi-r1:cuda:navatala_cfd_layer_norm_forward",
  5,
  kAbiArgs_cuda_navatala_cfd_layer_norm_forward
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_radius_count[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ref", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQuery", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRef", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "radiusIn", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_radius_count = {
  1,
  "navatala_cfd_radius_count",
  "cuda",
  "navatala_cfd_radius_count",
  "kernel:cuda:navatala_cfd_radius_count",
  "abi-r1:cuda:navatala_cfd_radius_count",
  "abi-r1:cuda:navatala_cfd_radius_count",
  6,
  kAbiArgs_cuda_navatala_cfd_radius_count
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scaled_dot_product[] = {
  { "q", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scores", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scaled_dot_product = {
  1,
  "navatala_cfd_scaled_dot_product",
  "cuda",
  "navatala_cfd_scaled_dot_product",
  "kernel:cuda:navatala_cfd_scaled_dot_product",
  "abi-r1:cuda:navatala_cfd_scaled_dot_product",
  "abi-r1:cuda:navatala_cfd_scaled_dot_product",
  6,
  kAbiArgs_cuda_navatala_cfd_scaled_dot_product
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_scatter_add_atomic[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "featureDim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_scatter_add_atomic = {
  1,
  "navatala_cfd_scatter_add_atomic",
  "cuda",
  "navatala_cfd_scatter_add_atomic",
  "kernel:cuda:navatala_cfd_scatter_add_atomic",
  "abi-r1:cuda:navatala_cfd_scatter_add_atomic",
  "abi-r1:cuda:navatala_cfd_scatter_add_atomic",
  5,
  kAbiArgs_cuda_navatala_cfd_scatter_add_atomic
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_spectral_complex_mul[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_spectral_complex_mul = {
  1,
  "navatala_cfd_spectral_complex_mul",
  "cuda",
  "navatala_cfd_spectral_complex_mul",
  "kernel:cuda:navatala_cfd_spectral_complex_mul",
  "abi-r1:cuda:navatala_cfd_spectral_complex_mul",
  "abi-r1:cuda:navatala_cfd_spectral_complex_mul",
  4,
  kAbiArgs_cuda_navatala_cfd_spectral_complex_mul
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_spectral_conv1d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPos", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNeg", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_spectral_conv1d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv1d_mode_sliced",
  "cuda",
  "navatala_cfd_spectral_conv1d_mode_sliced",
  "kernel:cuda:navatala_cfd_spectral_conv1d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv1d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv1d_mode_sliced",
  6,
  kAbiArgs_cuda_navatala_cfd_spectral_conv1d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_spectral_conv2d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPos", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNeg", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_spectral_conv2d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv2d_mode_sliced",
  "cuda",
  "navatala_cfd_spectral_conv2d_mode_sliced",
  "kernel:cuda:navatala_cfd_spectral_conv2d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv2d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv2d_mode_sliced",
  7,
  kAbiArgs_cuda_navatala_cfd_spectral_conv2d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_spectral_conv3d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPP", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNP", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPN", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNN", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftD", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_spectral_conv3d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv3d_mode_sliced",
  "cuda",
  "navatala_cfd_spectral_conv3d_mode_sliced",
  "kernel:cuda:navatala_cfd_spectral_conv3d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv3d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv3d_mode_sliced",
  10,
  kAbiArgs_cuda_navatala_cfd_spectral_conv3d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_spectral_conv4d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPPP", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNPP", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPNP", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNNP", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPPN", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNPN", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPNN", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNNN", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftT", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftD", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_spectral_conv4d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv4d_mode_sliced",
  "cuda",
  "navatala_cfd_spectral_conv4d_mode_sliced",
  "kernel:cuda:navatala_cfd_spectral_conv4d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv4d_mode_sliced",
  "abi-r1:cuda:navatala_cfd_spectral_conv4d_mode_sliced",
  15,
  kAbiArgs_cuda_navatala_cfd_spectral_conv4d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_compute_face_area_pair_weights_f32[] = {
  { "sfX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceWeight", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_compute_face_area_pair_weights_f32 = {
  1,
  "navatala_cfd_compute_face_area_pair_weights_f32",
  "cuda",
  "navatala_cfd_compute_face_area_pair_weights_f32",
  "kernel:cuda:navatala_cfd_compute_face_area_pair_weights_f32",
  "abi-r1:cuda:navatala_cfd_compute_face_area_pair_weights_f32",
  "abi-r1:cuda:navatala_cfd_compute_face_area_pair_weights_f32",
  6,
  kAbiArgs_cuda_navatala_cfd_compute_face_area_pair_weights_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_restrict_by_agglomeration_f32[] = {
  { "fineField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fineToCoarse", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFineCells", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseField", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_restrict_by_agglomeration_f32 = {
  1,
  "navatala_cfd_restrict_by_agglomeration_f32",
  "cuda",
  "navatala_cfd_restrict_by_agglomeration_f32",
  "kernel:cuda:navatala_cfd_restrict_by_agglomeration_f32",
  "abi-r1:cuda:navatala_cfd_restrict_by_agglomeration_f32",
  "abi-r1:cuda:navatala_cfd_restrict_by_agglomeration_f32",
  4,
  kAbiArgs_cuda_navatala_cfd_restrict_by_agglomeration_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_restrict_by_agglomeration_gather_f32[] = {
  { "fineField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseToFineOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseToFineList", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarseCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseField", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_restrict_by_agglomeration_gather_f32 = {
  1,
  "navatala_cfd_restrict_by_agglomeration_gather_f32",
  "cuda",
  "navatala_cfd_restrict_by_agglomeration_gather_f32",
  "kernel:cuda:navatala_cfd_restrict_by_agglomeration_gather_f32",
  "abi-r1:cuda:navatala_cfd_restrict_by_agglomeration_gather_f32",
  "abi-r1:cuda:navatala_cfd_restrict_by_agglomeration_gather_f32",
  5,
  kAbiArgs_cuda_navatala_cfd_restrict_by_agglomeration_gather_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_prolongate_by_agglomeration_f32[] = {
  { "coarseField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fineToCoarse", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFineCells", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "fineField", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_prolongate_by_agglomeration_f32 = {
  1,
  "navatala_cfd_prolongate_by_agglomeration_f32",
  "cuda",
  "navatala_cfd_prolongate_by_agglomeration_f32",
  "kernel:cuda:navatala_cfd_prolongate_by_agglomeration_f32",
  "abi-r1:cuda:navatala_cfd_prolongate_by_agglomeration_f32",
  "abi-r1:cuda:navatala_cfd_prolongate_by_agglomeration_f32",
  4,
  kAbiArgs_cuda_navatala_cfd_prolongate_by_agglomeration_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_diag_inv_apply_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "z", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_diag_inv_apply_f32 = {
  1,
  "navatala_cfd_diag_inv_apply_f32",
  "cuda",
  "navatala_cfd_diag_inv_apply_f32",
  "kernel:cuda:navatala_cfd_diag_inv_apply_f32",
  "abi-r1:cuda:navatala_cfd_diag_inv_apply_f32",
  "abi-r1:cuda:navatala_cfd_diag_inv_apply_f32",
  4,
  kAbiArgs_cuda_navatala_cfd_diag_inv_apply_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_csr_sp_m_v_mixed_f32_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_csr_sp_m_v_mixed_f32_f64 = {
  1,
  "navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "cuda",
  "navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "kernel:cuda:navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "abi-r1:cuda:navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "abi-r1:cuda:navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  6,
  kAbiArgs_cuda_navatala_cfd_csr_sp_m_v_mixed_f32_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_chebyshev_sweep_mixed_f32_f64[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "d", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "coeffB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dNew", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_chebyshev_sweep_mixed_f32_f64 = {
  1,
  "navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "cuda",
  "navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "kernel:cuda:navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "abi-r1:cuda:navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "abi-r1:cuda:navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  7,
  kAbiArgs_cuda_navatala_cfd_chebyshev_sweep_mixed_f32_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_ell_sp_m_v_mixed_f32_f64[] = {
  { "colIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxWidth", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_ell_sp_m_v_mixed_f32_f64 = {
  1,
  "navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "cuda",
  "navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "kernel:cuda:navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "abi-r1:cuda:navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "abi-r1:cuda:navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  6,
  kAbiArgs_cuda_navatala_cfd_ell_sp_m_v_mixed_f32_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_chebyshev_smooth_post_fused_mixed[] = {
  { "Az", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dPrev", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffA", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "coeffB", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "z", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dNew", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_chebyshev_smooth_post_fused_mixed = {
  1,
  "navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "cuda",
  "navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "kernel:cuda:navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "abi-r1:cuda:navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "abi-r1:cuda:navatala_cfd_chebyshev_smooth_post_fused_mixed",
  9,
  kAbiArgs_cuda_navatala_cfd_chebyshev_smooth_post_fused_mixed
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_proc_pack_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "localCells", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sendVals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_proc_pack_f32 = {
  1,
  "navatala_cfd_gamg_proc_pack_f32",
  "cuda",
  "navatala_cfd_gamg_proc_pack_f32",
  "kernel:cuda:navatala_cfd_gamg_proc_pack_f32",
  "abi-r1:cuda:navatala_cfd_gamg_proc_pack_f32",
  "abi-r1:cuda:navatala_cfd_gamg_proc_pack_f32",
  4,
  kAbiArgs_cuda_navatala_cfd_gamg_proc_pack_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_proc_add_f32[] = {
  { "localCells", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_proc_add_f32 = {
  1,
  "navatala_cfd_gamg_proc_add_f32",
  "cuda",
  "navatala_cfd_gamg_proc_add_f32",
  "kernel:cuda:navatala_cfd_gamg_proc_add_f32",
  "abi-r1:cuda:navatala_cfd_gamg_proc_add_f32",
  "abi-r1:cuda:navatala_cfd_gamg_proc_add_f32",
  5,
  kAbiArgs_cuda_navatala_cfd_gamg_proc_add_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_gamg_proc_add_gather_f32[] = {
  { "entryOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "entryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffs", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_gamg_proc_add_gather_f32 = {
  1,
  "navatala_cfd_gamg_proc_add_gather_f32",
  "cuda",
  "navatala_cfd_gamg_proc_add_gather_f32",
  "kernel:cuda:navatala_cfd_gamg_proc_add_gather_f32",
  "abi-r1:cuda:navatala_cfd_gamg_proc_add_gather_f32",
  "abi-r1:cuda:navatala_cfd_gamg_proc_add_gather_f32",
  6,
  kAbiArgs_cuda_navatala_cfd_gamg_proc_add_gather_f32
};

bool tryGetKernelAbiManifest_cuda_cfd(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "navatala_cfd_boundary_force_partials") {
    out = &kAbiManifest_cuda_navatala_cfd_boundary_force_partials;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_values") {
    out = &kAbiManifest_cuda_navatala_cfd_pack_owner_values;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_values_and_mask") {
    out = &kAbiManifest_cuda_navatala_cfd_scatter_values_and_mask;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_mu_grad_owner") {
    out = &kAbiManifest_cuda_navatala_cfd_pack_mu_grad_owner;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_rho_vec3") {
    out = &kAbiManifest_cuda_navatala_cfd_pack_owner_rho_vec3;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_vec3") {
    out = &kAbiManifest_cuda_navatala_cfd_pack_owner_vec3;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_vec3_mu") {
    out = &kAbiManifest_cuda_navatala_cfd_pack_owner_vec3_mu;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_mu_grad_ghost") {
    out = &kAbiManifest_cuda_navatala_cfd_scatter_mu_grad_ghost;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_rho_vec3_and_mask") {
    out = &kAbiManifest_cuda_navatala_cfd_scatter_rho_vec3_and_mask;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_vec3_and_mask") {
    out = &kAbiManifest_cuda_navatala_cfd_scatter_vec3_and_mask;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_vec3_mu_and_mask") {
    out = &kAbiManifest_cuda_navatala_cfd_scatter_vec3_mu_and_mask;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_add_vol_vector") {
    out = &kAbiManifest_cuda_navatala_cfd_add_vol_vector;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_div_rho_phi_u_explicit") {
    out = &kAbiManifest_cuda_navatala_cfd_div_rho_phi_u_explicit;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_hby_a_from_u_and_terms") {
    out = &kAbiManifest_cuda_navatala_cfd_hby_a_from_u_and_terms;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_apply_sp_to_hby_a") {
    out = &kAbiManifest_cuda_navatala_cfd_apply_sp_to_hby_a;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_scalar_explicit") {
    out = &kAbiManifest_cuda_navatala_cfd_laplacian_scalar_explicit;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_vector_explicit") {
    out = &kAbiManifest_cuda_navatala_cfd_laplacian_vector_explicit;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_predictor_hby_a_from_source") {
    out = &kAbiManifest_cuda_navatala_cfd_predictor_hby_a_from_source;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_predictor_inverse_diag") {
    out = &kAbiManifest_cuda_navatala_cfd_predictor_inverse_diag;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_predictor_phi_hby_a_fuse") {
    out = &kAbiManifest_cuda_navatala_cfd_predictor_phi_hby_a_fuse;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_axpy_cells") {
    out = &kAbiManifest_cuda_navatala_cfd_axpy_cells;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ddt_diagonal") {
    out = &kAbiManifest_cuda_navatala_cfd_ddt_diagonal;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_div_upwind_coeffs") {
    out = &kAbiManifest_cuda_navatala_cfd_div_upwind_coeffs;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_fvm_sp") {
    out = &kAbiManifest_cuda_navatala_cfd_fvm_sp;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_fvm_su_sp") {
    out = &kAbiManifest_cuda_navatala_cfd_fvm_su_sp;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_coeffs") {
    out = &kAbiManifest_cuda_navatala_cfd_laplacian_coeffs;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_coeffs_accum") {
    out = &kAbiManifest_cuda_navatala_cfd_laplacian_coeffs_accum;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_linear_upwind_correction_source") {
    out = &kAbiManifest_cuda_navatala_cfd_linear_upwind_correction_source;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_matrix_relax") {
    out = &kAbiManifest_cuda_navatala_cfd_matrix_relax;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_neg_sum_diag") {
    out = &kAbiManifest_cuda_navatala_cfd_neg_sum_diag;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_boundary_adjacent_cells") {
    out = &kAbiManifest_cuda_navatala_cfd_pack_boundary_adjacent_cells;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_jacobi_clamp_min") {
    out = &kAbiManifest_cuda_navatala_cfd_scalar_jacobi_clamp_min;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_jacobi_update") {
    out = &kAbiManifest_cuda_navatala_cfd_scalar_jacobi_update;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_ldu_mat_vec") {
    out = &kAbiManifest_cuda_navatala_cfd_scalar_ldu_mat_vec;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_ldu_multi_color_gs_sweep") {
    out = &kAbiManifest_cuda_navatala_cfd_scalar_ldu_multi_color_gs_sweep;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_sum_mag_off_diag") {
    out = &kAbiManifest_cuda_navatala_cfd_sum_mag_off_diag;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_axpy_in_place") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_axpy_in_place;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_compute_scale_factor") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_compute_scale_factor;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_dot_finalize") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_dot_finalize;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_negate_scalar") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_negate_scalar;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_scale_correction") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_scale_correction;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_dic_build") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_dic_build;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_dic_smooth") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_dic_smooth;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_coarse_jacobi") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_coarse_jacobi;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_coarse_matvec") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_coarse_matvec;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_coarse_rhs") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_coarse_rhs;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_find_best_match") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_find_best_match;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_pair_match") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_pair_match;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_prolong_add") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_prolong_add;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_renumber_aggregates") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_renumber_aggregates;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_renumber_propagate") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_renumber_propagate;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_resolve_chains") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_resolve_chains;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_restrict_sum") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_restrict_sum;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_strength_of_connection") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_strength_of_connection;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_zero_coarse") {
    out = &kAbiManifest_cuda_navatala_cfd_mg_zero_coarse;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_phi_correct") {
    out = &kAbiManifest_cuda_navatala_cfd_phi_correct;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_phi_correct_boundary") {
    out = &kAbiManifest_cuda_navatala_cfd_phi_correct_boundary;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_phi_correct_internal") {
    out = &kAbiManifest_cuda_navatala_cfd_phi_correct_internal;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_axpy_in_place") {
    out = &kAbiManifest_cuda_navatala_cfd_axpy_in_place;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_bc_dirichlet_face_flux") {
    out = &kAbiManifest_cuda_navatala_cfd_bc_dirichlet_face_flux;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_bc_sn_grad_face_flux") {
    out = &kAbiManifest_cuda_navatala_cfd_bc_sn_grad_face_flux;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_coeff_to_cf_in_place") {
    out = &kAbiManifest_cuda_navatala_cfd_coeff_to_cf_in_place;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_diag_from_cf") {
    out = &kAbiManifest_cuda_navatala_cfd_diag_from_cf;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dic_apply") {
    out = &kAbiManifest_cuda_navatala_cfd_dic_apply;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dic_build_reciprocal_d") {
    out = &kAbiManifest_cuda_navatala_cfd_dic_build_reciprocal_d;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dot_partials") {
    out = &kAbiManifest_cuda_navatala_cfd_dot_partials;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_face_flux") {
    out = &kAbiManifest_cuda_navatala_cfd_face_flux;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_inv_diag") {
    out = &kAbiManifest_cuda_navatala_cfd_inv_diag;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_from_face_flux") {
    out = &kAbiManifest_cuda_navatala_cfd_laplacian_from_face_flux;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mul_by_vol_in_place") {
    out = &kAbiManifest_cuda_navatala_cfd_mul_by_vol_in_place;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ref_add_ax") {
    out = &kAbiManifest_cuda_navatala_cfd_ref_add_ax;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ref_add_b") {
    out = &kAbiManifest_cuda_navatala_cfd_ref_add_b;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_rhs_sub") {
    out = &kAbiManifest_cuda_navatala_cfd_rhs_sub;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_shift_in_place") {
    out = &kAbiManifest_cuda_navatala_cfd_shift_in_place;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_sum_abs_partials") {
    out = &kAbiManifest_cuda_navatala_cfd_sum_abs_partials;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_upper_from_cf") {
    out = &kAbiManifest_cuda_navatala_cfd_upper_from_cf;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_xpay_in_place") {
    out = &kAbiManifest_cuda_navatala_cfd_xpay_in_place;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dot_finalize_partials") {
    out = &kAbiManifest_cuda_navatala_cfd_dot_finalize_partials;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_negate_scalar") {
    out = &kAbiManifest_cuda_navatala_cfd_negate_scalar;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pcg_compute_alpha") {
    out = &kAbiManifest_cuda_navatala_cfd_pcg_compute_alpha;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pcg_compute_beta") {
    out = &kAbiManifest_cuda_navatala_cfd_pcg_compute_beta;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_exact_mat_vec") {
    out = &kAbiManifest_cuda_navatala_cfd_exact_mat_vec;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_exact_mat_vec_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_exact_mat_vec_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_face_flux_boundary") {
    out = &kAbiManifest_cuda_navatala_cfd_face_flux_boundary;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_face_flux_internal") {
    out = &kAbiManifest_cuda_navatala_cfd_face_flux_internal;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pressure_face_flux_correction_corrected") {
    out = &kAbiManifest_cuda_navatala_cfd_pressure_face_flux_correction_corrected;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pressure_sn_grad") {
    out = &kAbiManifest_cuda_navatala_cfd_pressure_sn_grad;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_u_correct") {
    out = &kAbiManifest_cuda_navatala_cfd_u_correct;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_u_correct_reconstruct") {
    out = &kAbiManifest_cuda_navatala_cfd_u_correct_reconstruct;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_grad_alpha_cell") {
    out = &kAbiManifest_cuda_navatala_cfd_grad_alpha_cell;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_grad_alpha_cell_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_grad_alpha_cell_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_n_hatf_all") {
    out = &kAbiManifest_cuda_navatala_cfd_n_hatf_all;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_n_hatf_all_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_n_hatf_all_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_stf_all_faces") {
    out = &kAbiManifest_cuda_navatala_cfd_stf_all_faces;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_stf_all_faces_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_stf_all_faces_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_k_omega_s_s_t_blending") {
    out = &kAbiManifest_cuda_navatala_cfd_k_omega_s_s_t_blending;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_k_omega_s_s_t_nut_compute") {
    out = &kAbiManifest_cuda_navatala_cfd_k_omega_s_s_t_nut_compute;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_k_omega_s_s_t_sources") {
    out = &kAbiManifest_cuda_navatala_cfd_k_omega_s_s_t_sources;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mu_eff_from_rho_nu_eff") {
    out = &kAbiManifest_cuda_navatala_cfd_mu_eff_from_rho_nu_eff;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix") {
    out = &kAbiManifest_cuda_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_turbulence_add_explicit_source_device") {
    out = &kAbiManifest_cuda_navatala_cfd_turbulence_add_explicit_source_device;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_turbulence_dirichlet_cell_source") {
    out = &kAbiManifest_cuda_navatala_cfd_turbulence_dirichlet_cell_source;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_turbulence_dirichlet_face_elimination") {
    out = &kAbiManifest_cuda_navatala_cfd_turbulence_dirichlet_face_elimination;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_two_mag_sqr_symm_grad_u") {
    out = &kAbiManifest_cuda_navatala_cfd_two_mag_sqr_symm_grad_u;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_attention_row_softmax") {
    out = &kAbiManifest_cuda_navatala_cfd_attention_row_softmax;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_attention_value_projection") {
    out = &kAbiManifest_cuda_navatala_cfd_attention_value_projection;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gather_mean") {
    out = &kAbiManifest_cuda_navatala_cfd_gather_mean;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_layer_norm_forward") {
    out = &kAbiManifest_cuda_navatala_cfd_layer_norm_forward;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_radius_count") {
    out = &kAbiManifest_cuda_navatala_cfd_radius_count;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scaled_dot_product") {
    out = &kAbiManifest_cuda_navatala_cfd_scaled_dot_product;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_add_atomic") {
    out = &kAbiManifest_cuda_navatala_cfd_scatter_add_atomic;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_complex_mul") {
    out = &kAbiManifest_cuda_navatala_cfd_spectral_complex_mul;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv1d_mode_sliced") {
    out = &kAbiManifest_cuda_navatala_cfd_spectral_conv1d_mode_sliced;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv2d_mode_sliced") {
    out = &kAbiManifest_cuda_navatala_cfd_spectral_conv2d_mode_sliced;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv3d_mode_sliced") {
    out = &kAbiManifest_cuda_navatala_cfd_spectral_conv3d_mode_sliced;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv4d_mode_sliced") {
    out = &kAbiManifest_cuda_navatala_cfd_spectral_conv4d_mode_sliced;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_compute_face_area_pair_weights_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_compute_face_area_pair_weights_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_restrict_by_agglomeration_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_restrict_by_agglomeration_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_restrict_by_agglomeration_gather_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_restrict_by_agglomeration_gather_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_prolongate_by_agglomeration_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_prolongate_by_agglomeration_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_diag_inv_apply_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_diag_inv_apply_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_csr_sp_m_v_mixed_f32_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_csr_sp_m_v_mixed_f32_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_chebyshev_sweep_mixed_f32_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_chebyshev_sweep_mixed_f32_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ell_sp_m_v_mixed_f32_f64") {
    out = &kAbiManifest_cuda_navatala_cfd_ell_sp_m_v_mixed_f32_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_chebyshev_smooth_post_fused_mixed") {
    out = &kAbiManifest_cuda_navatala_cfd_chebyshev_smooth_post_fused_mixed;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_proc_pack_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_proc_pack_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_proc_add_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_proc_add_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_proc_add_gather_f32") {
    out = &kAbiManifest_cuda_navatala_cfd_gamg_proc_add_gather_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_cfd(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "navatala_cfd_boundary_force_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_boundary_force_partials";
    std::string_view sv(k_cuda_navatala_cfd_boundary_force_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_values") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pack_owner_values";
    std::string_view sv(k_cuda_navatala_cfd_pack_owner_values);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_values_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scatter_values_and_mask";
    std::string_view sv(k_cuda_navatala_cfd_scatter_values_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_mu_grad_owner") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pack_mu_grad_owner";
    std::string_view sv(k_cuda_navatala_cfd_pack_mu_grad_owner);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_rho_vec3") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pack_owner_rho_vec3";
    std::string_view sv(k_cuda_navatala_cfd_pack_owner_rho_vec3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_vec3") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pack_owner_vec3";
    std::string_view sv(k_cuda_navatala_cfd_pack_owner_vec3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_owner_vec3_mu") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pack_owner_vec3_mu";
    std::string_view sv(k_cuda_navatala_cfd_pack_owner_vec3_mu);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_mu_grad_ghost") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scatter_mu_grad_ghost";
    std::string_view sv(k_cuda_navatala_cfd_scatter_mu_grad_ghost);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_rho_vec3_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scatter_rho_vec3_and_mask";
    std::string_view sv(k_cuda_navatala_cfd_scatter_rho_vec3_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_vec3_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scatter_vec3_and_mask";
    std::string_view sv(k_cuda_navatala_cfd_scatter_vec3_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_vec3_mu_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scatter_vec3_mu_and_mask";
    std::string_view sv(k_cuda_navatala_cfd_scatter_vec3_mu_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_add_vol_vector") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_add_vol_vector";
    std::string_view sv(k_cuda_navatala_cfd_add_vol_vector);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_div_rho_phi_u_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_div_rho_phi_u_explicit";
    std::string_view sv(k_cuda_navatala_cfd_div_rho_phi_u_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_hby_a_from_u_and_terms") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_hby_a_from_u_and_terms";
    std::string_view sv(k_cuda_navatala_cfd_hby_a_from_u_and_terms);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_apply_sp_to_hby_a") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_apply_sp_to_hby_a";
    std::string_view sv(k_cuda_navatala_cfd_apply_sp_to_hby_a);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_scalar_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_laplacian_scalar_explicit";
    std::string_view sv(k_cuda_navatala_cfd_laplacian_scalar_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_vector_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_laplacian_vector_explicit";
    std::string_view sv(k_cuda_navatala_cfd_laplacian_vector_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_predictor_hby_a_from_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_predictor_hby_a_from_source";
    std::string_view sv(k_cuda_navatala_cfd_predictor_hby_a_from_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_predictor_inverse_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_predictor_inverse_diag";
    std::string_view sv(k_cuda_navatala_cfd_predictor_inverse_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_predictor_phi_hby_a_fuse") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_predictor_phi_hby_a_fuse";
    std::string_view sv(k_cuda_navatala_cfd_predictor_phi_hby_a_fuse);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_axpy_cells") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_axpy_cells";
    std::string_view sv(k_cuda_navatala_cfd_axpy_cells);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ddt_diagonal") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_ddt_diagonal";
    std::string_view sv(k_cuda_navatala_cfd_ddt_diagonal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_div_upwind_coeffs") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_div_upwind_coeffs";
    std::string_view sv(k_cuda_navatala_cfd_div_upwind_coeffs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_fvm_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_fvm_sp";
    std::string_view sv(k_cuda_navatala_cfd_fvm_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_fvm_su_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_fvm_su_sp";
    std::string_view sv(k_cuda_navatala_cfd_fvm_su_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_coeffs") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_laplacian_coeffs";
    std::string_view sv(k_cuda_navatala_cfd_laplacian_coeffs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_coeffs_accum") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_laplacian_coeffs_accum";
    std::string_view sv(k_cuda_navatala_cfd_laplacian_coeffs_accum);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_linear_upwind_correction_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_linear_upwind_correction_source";
    std::string_view sv(k_cuda_navatala_cfd_linear_upwind_correction_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_matrix_relax") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_matrix_relax";
    std::string_view sv(k_cuda_navatala_cfd_matrix_relax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_neg_sum_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_neg_sum_diag";
    std::string_view sv(k_cuda_navatala_cfd_neg_sum_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pack_boundary_adjacent_cells") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pack_boundary_adjacent_cells";
    std::string_view sv(k_cuda_navatala_cfd_pack_boundary_adjacent_cells);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_jacobi_clamp_min") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scalar_jacobi_clamp_min";
    std::string_view sv(k_cuda_navatala_cfd_scalar_jacobi_clamp_min);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_jacobi_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scalar_jacobi_update";
    std::string_view sv(k_cuda_navatala_cfd_scalar_jacobi_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_ldu_mat_vec") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scalar_ldu_mat_vec";
    std::string_view sv(k_cuda_navatala_cfd_scalar_ldu_mat_vec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scalar_ldu_multi_color_gs_sweep") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scalar_ldu_multi_color_gs_sweep";
    std::string_view sv(k_cuda_navatala_cfd_scalar_ldu_multi_color_gs_sweep);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_sum_mag_off_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_sum_mag_off_diag";
    std::string_view sv(k_cuda_navatala_cfd_sum_mag_off_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_axpy_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_axpy_in_place";
    std::string_view sv(k_cuda_navatala_cfd_gamg_axpy_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_compute_scale_factor") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_compute_scale_factor";
    std::string_view sv(k_cuda_navatala_cfd_gamg_compute_scale_factor);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_dot_finalize") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_dot_finalize";
    std::string_view sv(k_cuda_navatala_cfd_gamg_dot_finalize);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_negate_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_negate_scalar";
    std::string_view sv(k_cuda_navatala_cfd_gamg_negate_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_scale_correction") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_scale_correction";
    std::string_view sv(k_cuda_navatala_cfd_gamg_scale_correction);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_dic_build") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_dic_build";
    std::string_view sv(k_cuda_navatala_cfd_mg_dic_build);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_dic_smooth") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_dic_smooth";
    std::string_view sv(k_cuda_navatala_cfd_mg_dic_smooth);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_coarse_jacobi") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_coarse_jacobi";
    std::string_view sv(k_cuda_navatala_cfd_mg_coarse_jacobi);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_coarse_matvec") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_coarse_matvec";
    std::string_view sv(k_cuda_navatala_cfd_mg_coarse_matvec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_coarse_rhs") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_coarse_rhs";
    std::string_view sv(k_cuda_navatala_cfd_mg_coarse_rhs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_find_best_match") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_find_best_match";
    std::string_view sv(k_cuda_navatala_cfd_mg_find_best_match);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_pair_match") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_pair_match";
    std::string_view sv(k_cuda_navatala_cfd_mg_pair_match);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_prolong_add") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_prolong_add";
    std::string_view sv(k_cuda_navatala_cfd_mg_prolong_add);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_renumber_aggregates") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_renumber_aggregates";
    std::string_view sv(k_cuda_navatala_cfd_mg_renumber_aggregates);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_renumber_propagate") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_renumber_propagate";
    std::string_view sv(k_cuda_navatala_cfd_mg_renumber_propagate);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_resolve_chains") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_resolve_chains";
    std::string_view sv(k_cuda_navatala_cfd_mg_resolve_chains);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_restrict_sum") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_restrict_sum";
    std::string_view sv(k_cuda_navatala_cfd_mg_restrict_sum);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_strength_of_connection") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_strength_of_connection";
    std::string_view sv(k_cuda_navatala_cfd_mg_strength_of_connection);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mg_zero_coarse") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mg_zero_coarse";
    std::string_view sv(k_cuda_navatala_cfd_mg_zero_coarse);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_phi_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_phi_correct";
    std::string_view sv(k_cuda_navatala_cfd_phi_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_phi_correct_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_phi_correct_boundary";
    std::string_view sv(k_cuda_navatala_cfd_phi_correct_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_phi_correct_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_phi_correct_internal";
    std::string_view sv(k_cuda_navatala_cfd_phi_correct_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_axpy_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_axpy_in_place";
    std::string_view sv(k_cuda_navatala_cfd_axpy_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_bc_dirichlet_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_bc_dirichlet_face_flux";
    std::string_view sv(k_cuda_navatala_cfd_bc_dirichlet_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_bc_sn_grad_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_bc_sn_grad_face_flux";
    std::string_view sv(k_cuda_navatala_cfd_bc_sn_grad_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_coeff_to_cf_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_coeff_to_cf_in_place";
    std::string_view sv(k_cuda_navatala_cfd_coeff_to_cf_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_diag_from_cf") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_diag_from_cf";
    std::string_view sv(k_cuda_navatala_cfd_diag_from_cf);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dic_apply") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_dic_apply";
    std::string_view sv(k_cuda_navatala_cfd_dic_apply);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dic_build_reciprocal_d") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_dic_build_reciprocal_d";
    std::string_view sv(k_cuda_navatala_cfd_dic_build_reciprocal_d);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dot_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_dot_partials";
    std::string_view sv(k_cuda_navatala_cfd_dot_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_face_flux";
    std::string_view sv(k_cuda_navatala_cfd_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_inv_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_inv_diag";
    std::string_view sv(k_cuda_navatala_cfd_inv_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_laplacian_from_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_laplacian_from_face_flux";
    std::string_view sv(k_cuda_navatala_cfd_laplacian_from_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mul_by_vol_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mul_by_vol_in_place";
    std::string_view sv(k_cuda_navatala_cfd_mul_by_vol_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ref_add_ax") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_ref_add_ax";
    std::string_view sv(k_cuda_navatala_cfd_ref_add_ax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ref_add_b") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_ref_add_b";
    std::string_view sv(k_cuda_navatala_cfd_ref_add_b);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_rhs_sub") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_rhs_sub";
    std::string_view sv(k_cuda_navatala_cfd_rhs_sub);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_shift_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_shift_in_place";
    std::string_view sv(k_cuda_navatala_cfd_shift_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_sum_abs_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_sum_abs_partials";
    std::string_view sv(k_cuda_navatala_cfd_sum_abs_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_upper_from_cf") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_upper_from_cf";
    std::string_view sv(k_cuda_navatala_cfd_upper_from_cf);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_xpay_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_xpay_in_place";
    std::string_view sv(k_cuda_navatala_cfd_xpay_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_dot_finalize_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_dot_finalize_partials";
    std::string_view sv(k_cuda_navatala_cfd_dot_finalize_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_negate_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_negate_scalar";
    std::string_view sv(k_cuda_navatala_cfd_negate_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pcg_compute_alpha") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pcg_compute_alpha";
    std::string_view sv(k_cuda_navatala_cfd_pcg_compute_alpha);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pcg_compute_beta") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pcg_compute_beta";
    std::string_view sv(k_cuda_navatala_cfd_pcg_compute_beta);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_exact_mat_vec") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_exact_mat_vec";
    std::string_view sv(k_cuda_navatala_cfd_exact_mat_vec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_exact_mat_vec_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_exact_mat_vec_f64";
    std::string_view sv(k_cuda_navatala_cfd_exact_mat_vec_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_face_flux_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_face_flux_boundary";
    std::string_view sv(k_cuda_navatala_cfd_face_flux_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_face_flux_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_face_flux_internal";
    std::string_view sv(k_cuda_navatala_cfd_face_flux_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pressure_face_flux_correction_corrected") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pressure_face_flux_correction_corrected";
    std::string_view sv(k_cuda_navatala_cfd_pressure_face_flux_correction_corrected);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_pressure_sn_grad") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_pressure_sn_grad";
    std::string_view sv(k_cuda_navatala_cfd_pressure_sn_grad);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_u_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_u_correct";
    std::string_view sv(k_cuda_navatala_cfd_u_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_u_correct_reconstruct") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_u_correct_reconstruct";
    std::string_view sv(k_cuda_navatala_cfd_u_correct_reconstruct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_grad_alpha_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_grad_alpha_cell";
    std::string_view sv(k_cuda_navatala_cfd_grad_alpha_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_grad_alpha_cell_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_grad_alpha_cell_f64";
    std::string_view sv(k_cuda_navatala_cfd_grad_alpha_cell_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_n_hatf_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_n_hatf_all";
    std::string_view sv(k_cuda_navatala_cfd_n_hatf_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_n_hatf_all_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_n_hatf_all_f64";
    std::string_view sv(k_cuda_navatala_cfd_n_hatf_all_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_stf_all_faces") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_stf_all_faces";
    std::string_view sv(k_cuda_navatala_cfd_stf_all_faces);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_stf_all_faces_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_stf_all_faces_f64";
    std::string_view sv(k_cuda_navatala_cfd_stf_all_faces_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_k_omega_s_s_t_blending") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_blending";
    std::string_view sv(k_cuda_navatala_cfd_k_omega_s_s_t_blending);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_k_omega_s_s_t_nut_compute") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_nut_compute";
    std::string_view sv(k_cuda_navatala_cfd_k_omega_s_s_t_nut_compute);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_k_omega_s_s_t_sources") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_sources";
    std::string_view sv(k_cuda_navatala_cfd_k_omega_s_s_t_sources);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mu_eff_from_rho_nu_eff") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mu_eff_from_rho_nu_eff";
    std::string_view sv(k_cuda_navatala_cfd_mu_eff_from_rho_nu_eff);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix";
    std::string_view sv(k_cuda_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_turbulence_add_explicit_source_device") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_turbulence_add_explicit_source_device";
    std::string_view sv(k_cuda_navatala_cfd_turbulence_add_explicit_source_device);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_turbulence_dirichlet_cell_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_turbulence_dirichlet_cell_source";
    std::string_view sv(k_cuda_navatala_cfd_turbulence_dirichlet_cell_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_turbulence_dirichlet_face_elimination") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_turbulence_dirichlet_face_elimination";
    std::string_view sv(k_cuda_navatala_cfd_turbulence_dirichlet_face_elimination);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_two_mag_sqr_symm_grad_u") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_two_mag_sqr_symm_grad_u";
    std::string_view sv(k_cuda_navatala_cfd_two_mag_sqr_symm_grad_u);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_attention_row_softmax") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_attention_row_softmax";
    std::string_view sv(k_cuda_navatala_cfd_attention_row_softmax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_attention_value_projection") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_attention_value_projection";
    std::string_view sv(k_cuda_navatala_cfd_attention_value_projection);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gather_mean") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gather_mean";
    std::string_view sv(k_cuda_navatala_cfd_gather_mean);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_layer_norm_forward") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_layer_norm_forward";
    std::string_view sv(k_cuda_navatala_cfd_layer_norm_forward);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_radius_count") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_radius_count";
    std::string_view sv(k_cuda_navatala_cfd_radius_count);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scaled_dot_product") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scaled_dot_product";
    std::string_view sv(k_cuda_navatala_cfd_scaled_dot_product);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_scatter_add_atomic") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_scatter_add_atomic";
    std::string_view sv(k_cuda_navatala_cfd_scatter_add_atomic);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_complex_mul") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_spectral_complex_mul";
    std::string_view sv(k_cuda_navatala_cfd_spectral_complex_mul);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv1d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_spectral_conv1d_mode_sliced";
    std::string_view sv(k_cuda_navatala_cfd_spectral_conv1d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv2d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_spectral_conv2d_mode_sliced";
    std::string_view sv(k_cuda_navatala_cfd_spectral_conv2d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv3d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_spectral_conv3d_mode_sliced";
    std::string_view sv(k_cuda_navatala_cfd_spectral_conv3d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_spectral_conv4d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_spectral_conv4d_mode_sliced";
    std::string_view sv(k_cuda_navatala_cfd_spectral_conv4d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_compute_face_area_pair_weights_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_compute_face_area_pair_weights_f32";
    std::string_view sv(k_cuda_navatala_cfd_compute_face_area_pair_weights_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_restrict_by_agglomeration_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_restrict_by_agglomeration_f32";
    std::string_view sv(k_cuda_navatala_cfd_restrict_by_agglomeration_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_restrict_by_agglomeration_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_restrict_by_agglomeration_gather_f32";
    std::string_view sv(k_cuda_navatala_cfd_restrict_by_agglomeration_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_prolongate_by_agglomeration_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_prolongate_by_agglomeration_f32";
    std::string_view sv(k_cuda_navatala_cfd_prolongate_by_agglomeration_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_diag_inv_apply_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_diag_inv_apply_f32";
    std::string_view sv(k_cuda_navatala_cfd_diag_inv_apply_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_csr_sp_m_v_mixed_f32_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_csr_sp_m_v_mixed_f32_f64";
    std::string_view sv(k_cuda_navatala_cfd_csr_sp_m_v_mixed_f32_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_chebyshev_sweep_mixed_f32_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_chebyshev_sweep_mixed_f32_f64";
    std::string_view sv(k_cuda_navatala_cfd_chebyshev_sweep_mixed_f32_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_ell_sp_m_v_mixed_f32_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_ell_sp_m_v_mixed_f32_f64";
    std::string_view sv(k_cuda_navatala_cfd_ell_sp_m_v_mixed_f32_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_chebyshev_smooth_post_fused_mixed") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_chebyshev_smooth_post_fused_mixed";
    std::string_view sv(k_cuda_navatala_cfd_chebyshev_smooth_post_fused_mixed);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_proc_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_proc_pack_f32";
    std::string_view sv(k_cuda_navatala_cfd_gamg_proc_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_proc_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_proc_add_f32";
    std::string_view sv(k_cuda_navatala_cfd_gamg_proc_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_gamg_proc_add_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_gamg_proc_add_gather_f32";
    std::string_view sv(k_cuda_navatala_cfd_gamg_proc_add_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

