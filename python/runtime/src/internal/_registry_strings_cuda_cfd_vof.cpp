// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_cfd_vof` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_navatala_cfd_vof_alpha_phi_all = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_alpha_phi_all(const float* alphaPhiInt, const float* phiAll, const float* alphaF, const int* params, float* alphaPhiAllOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      alphaPhiAllOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = alphaPhiInt[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    } else {
      float a = alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (a < __uint_as_float(0x00000000u)) {
        a = __uint_as_float(0x00000000u);
      }
      if (a > __uint_as_float(0x3f800000u)) {
        a = __uint_as_float(0x3f800000u);
      }
      alphaPhiAllOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * a);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_alpha_phi_int = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_alpha_phi_int(const float* alpha, const float* phi, const int* owner, const int* nei, const float* phir, const float* alphaF, const int* faceParams, float* outAlphaPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float a = __uint_as_float(0x00000000u);
    if (ph >= __uint_as_float(0x00000000u)) {
      a = alpha[o];
    } else {
      a = alpha[n];
    }
    if (a < __uint_as_float(0x00000000u)) {
      a = __uint_as_float(0x00000000u);
    }
    if (a > __uint_as_float(0x3f800000u)) {
      a = __uint_as_float(0x3f800000u);
    }
    float ac = alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (ac < __uint_as_float(0x00000000u)) {
      ac = __uint_as_float(0x00000000u);
    }
    if (ac > __uint_as_float(0x3f800000u)) {
      ac = __uint_as_float(0x3f800000u);
    }
    float oneMinus = (__uint_as_float(0x3f800000u) - ac);
    float comp = (ac * oneMinus);
    outAlphaPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((ph * a) + (phir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * comp));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_average_face_scalar_to_cell = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_average_face_scalar_to_cell(const float* alphaF, const float* magSf, const unsigned int* offsets, const unsigned int* faceIdx, const float* sumMag, const unsigned int* counts, float* outAlpha) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = __uint_as_float(0x00000000u);
    unsigned int beg = offsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    unsigned int end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      unsigned int f = faceIdx[k];
      s = (s + (magSf[((int)(f))] * alphaF[((int)(f))]));
    }
    float d = sumMag[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (d < __uint_as_float(0x1e3ce508u)) {
      outAlpha[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (s / __uint_as_float(0x1e3ce508u));
    } else {
      outAlpha[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (s / d);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_interp_scalar_face_all = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_interp_scalar_face_all(const float* alpha, const int* owner, const int* nei, const float* w, const float* bcAlphaF, const unsigned int* params, float* alphaF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) < ((int)(params[1]))) {
      int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      int n = nei[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float wf = w[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float wc = (wf * ((float)((wf > __uint_as_float(0x00000000u)))));
      float a = ((wc * alpha[o]) + ((__uint_as_float(0x3f800000u) - wc) * alpha[n]));
      float ac = (a * ((float)((a > __uint_as_float(0x00000000u)))));
      if (ac > __uint_as_float(0x3f800000u)) {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x3f800000u);
      } else {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ac;
      }
    } else {
      float a = bcAlphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float ac = (a * ((float)((a > __uint_as_float(0x00000000u)))));
      if (ac > __uint_as_float(0x3f800000u)) {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x3f800000u);
      } else {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ac;
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_alpha_face_average_internal = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_alpha_face_average_internal(const float* alpha, const int* owner, const int* neighbour, float* alphaF, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float a = (__uint_as_float(0x3f000000u) * (alpha[o] + alpha[n]));
    if (a < __uint_as_float(0x00000000u)) {
      a = __uint_as_float(0x00000000u);
    }
    if (a > __uint_as_float(0x3f800000u)) {
      a = __uint_as_float(0x3f800000u);
    }
    alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = a;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_alpha_update = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_alpha_update(const float* alpha, const float* divA, const float* paramsF, const int* counts, const float* rSubDeltaT, float* alphaNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float a = (alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (divA[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] / rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    if (a < __uint_as_float(0x00000000u)) {
      a = __uint_as_float(0x00000000u);
    }
    if (a > __uint_as_float(0x3f800000u)) {
      a = __uint_as_float(0x3f800000u);
    }
    alphaNew[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = a;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_cmules_apply_correction = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_cmules_apply_correction(const float* phiCorr, const float* lambda, const int* counts, const float* paramsF, float* phiCorrLimited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    phiCorrLimited[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_cmules_boundary_face_update = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_cmules_boundary_face_update(const float* phi, const float* phiCorr, float* lambda, const int* ownerAll, const float* lambdam, const float* lambdap, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2]) {
      return;
    } else {
      float pc = phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float combined = (phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + pc);
      float eps2 = (paramsF[1] * paramsF[1]);
      if (combined <= eps2) {
        return;
      } else {
        int o = ownerAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float lim = lambdam[o];
        if (pc > __uint_as_float(0x00000000u)) {
          lim = lambdap[o];
        }
        float cur = lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float _out = cur;
        if (_out > lim) {
          _out = lim;
        }
        lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = _out;
      }
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_cmules_correct = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_cmules_correct(const float* alphaOld, const float* rSubDeltaT, const float* sp, const float* su, const float* phiCorrLimited, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* counts, const float* paramsF, float* alphaNext) {
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
    float acc = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float term = (sign[k] * phiCorrLimited[f]);
      acc = (acc + term);
    }
    float divCorr = (acc / vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    float rdt = rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float spv = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float suv = su[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float den = (rdt - spv);
    alphaNext[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((((alphaOld[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * rdt) + suv) - divCorr) / den);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_cmules_limiter_corr_prepare = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_cmules_limiter_corr_prepare(const float* alpha, const float* alphaF, const float* phiCorr, const int* offsets, const int* faceIdx, const float* sign, const int* owner, const int* nei, const float* vol, const float* rSubDeltaT, const float* sp, const float* su, const float* psiMax, const float* psiMin, const int* counts, const float* paramsF, float* psiMaxCap, float* psiMinCap, float* sumPhip, float* mSumPhim) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float psiMaxCell = psiMax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float psiMinCell = psiMin[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float maxN = psiMinCell;
    float minN = psiMaxCell;
    float spSum = __uint_as_float(0x00000000u);
    float smSum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = __uint_as_float(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > __uint_as_float(0x00000000u)) {
          nbr = nei[f];
        }
        v = alpha[nbr];
      } else {
        v = alphaF[f];
      }
      if (v > maxN) {
        maxN = v;
      }
      if (v < minN) {
        minN = v;
      }
      float pc = phiCorr[f];
      if (f < counts[2]) {
        if (s > __uint_as_float(0x00000000u)) {
          if (pc > __uint_as_float(0x00000000u)) {
            spSum = (spSum + pc);
          } else {
            smSum = (smSum - pc);
          }
        } else {
          if (pc > __uint_as_float(0x00000000u)) {
            smSum = (smSum + pc);
          } else {
            spSum = (spSum - pc);
          }
        }
      } else {
        if (pc > __uint_as_float(0x00000000u)) {
          spSum = (spSum + pc);
        } else {
          smSum = (smSum - pc);
        }
      }
    }
    float range = (psiMaxCell - psiMinCell);
    maxN = (maxN + (paramsF[2] * range));
    if (maxN > psiMaxCell) {
      maxN = psiMaxCell;
    }
    minN = (minN - (paramsF[2] * range));
    if (minN < psiMinCell) {
      minN = psiMinCell;
    }
    if (paramsF[3] > __uint_as_float(0x00000000u)) {
      float omSmooth = (__uint_as_float(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > psiMaxCell) {
        maxN = psiMaxCell;
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < psiMinCell) {
        minN = psiMinCell;
      }
    }
    sumPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = spSum;
    mSumPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = smSum;
    float V = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float rdt = rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float spv = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float suv = su[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float diag = (rdt - spv);
    psiMaxCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (V * (((diag * maxN) - suv) - (rdt * psi)));
    psiMinCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (V * ((suv - (diag * minN)) + (rdt * psi)));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_mules_apply = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_mules_apply(const float* phiBD, const float* phiCorr, const float* lambda, const int* counts, const float* paramsF, float* alphaPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    alphaPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phiBD[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_mules_cell_lambda = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_mules_cell_lambda(const float* psiMaxCap, const float* psiMinCap, const float* sumPhip, const float* mSumPhim, const float* sumlPhip, const float* mSumlPhim, const int* counts, const float* paramsF, float* lambdam, float* lambdap) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float ld = ((sumlPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + psiMaxCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) / (mSumPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + paramsF[1]));
    float lp = ((mSumlPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + psiMinCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) / (sumPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + paramsF[1]));
    if (ld < __uint_as_float(0x00000000u)) {
      ld = __uint_as_float(0x00000000u);
    }
    if (ld > __uint_as_float(0x3f800000u)) {
      ld = __uint_as_float(0x3f800000u);
    }
    if (lp < __uint_as_float(0x00000000u)) {
      lp = __uint_as_float(0x00000000u);
    }
    if (lp > __uint_as_float(0x3f800000u)) {
      lp = __uint_as_float(0x3f800000u);
    }
    lambdam[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ld;
    lambdap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = lp;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_mules_cell_sums = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_mules_cell_sums(const float* phiCorr, const float* lambda, const int* offsets, const int* faceIdx, const float* sign, const int* counts, const float* paramsF, float* sumlPhip, float* mSumlPhim) {
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
    float sp = __uint_as_float(0x00000000u);
    float sm = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float lpc = (lambda[f] * phiCorr[f]);
      float s = sign[k];
      if (f < counts[2]) {
        if (s > __uint_as_float(0x00000000u)) {
          if (lpc > __uint_as_float(0x00000000u)) {
            sp = (sp + lpc);
          } else {
            sm = (sm + (-lpc));
          }
        } else {
          if (lpc > __uint_as_float(0x00000000u)) {
            sm = (sm + lpc);
          } else {
            sp = (sp + (-lpc));
          }
        }
      } else {
        if (lpc > __uint_as_float(0x00000000u)) {
          sp = (sp + lpc);
        } else {
          sm = (sm + (-lpc));
        }
      }
    }
    sumlPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sp;
    mSumlPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sm;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_mules_face_update = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_mules_face_update(const float* phiCorr, float* lambda, const int* owner, const int* nei, const float* lambdam, const float* lambdap, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[2])) > 0 ? ((int)(counts[2])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[2]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[2]) {
    return;
  } else {
    float pc = phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float a0 = lambdam[o];
    float b0 = lambdap[n];
    float lim = a0;
    if (lim > b0) {
      lim = b0;
    }
    if (pc > __uint_as_float(0x00000000u)) {
      float a1 = lambdap[o];
      float b1 = lambdam[n];
      float lim1 = a1;
      if (lim1 > b1) {
        lim1 = b1;
      }
      lim = lim1;
    }
    float cur = lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = _out;
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_mules_fill_lambda = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_mules_fill_lambda(const int* counts, const float* paramsF, float* lambda) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    lambda[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x3f800000u);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_mules_init = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_mules_init(const float* alpha, const float* alphaF, const float* phiBD, const float* phiCorr, const int* offsets, const int* faceIdx, const float* sign, const int* owner, const int* nei, const float* vol, const int* counts, const float* paramsF, const float* rSubDeltaT, float* psiMaxCap, float* psiMinCap, float* sumPhip, float* mSumPhim) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (psi < __uint_as_float(0x00000000u)) {
      psi = __uint_as_float(0x00000000u);
    }
    if (psi > __uint_as_float(0x3f800000u)) {
      psi = __uint_as_float(0x3f800000u);
    }
    float maxN = __uint_as_float(0x00000000u);
    float minN = __uint_as_float(0x3f800000u);
    float sumPhiBD = __uint_as_float(0x00000000u);
    float sp = __uint_as_float(0x00000000u);
    float sm = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = __uint_as_float(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > __uint_as_float(0x00000000u)) {
          nbr = nei[f];
        }
        v = alpha[nbr];
      } else {
        v = alphaF[f];
      }
      if (v < __uint_as_float(0x00000000u)) {
        v = __uint_as_float(0x00000000u);
      }
      if (v > __uint_as_float(0x3f800000u)) {
        v = __uint_as_float(0x3f800000u);
      }
      if (v > maxN) {
        maxN = v;
      }
      if (v < minN) {
        minN = v;
      }
      float termBD = (s * phiBD[f]);
      sumPhiBD = (sumPhiBD + termBD);
      float pc = phiCorr[f];
      if (f < counts[2]) {
        if (s > __uint_as_float(0x00000000u)) {
          if (pc > __uint_as_float(0x00000000u)) {
            sp = (sp + pc);
          } else {
            sm = (sm + (-pc));
          }
        } else {
          if (pc > __uint_as_float(0x00000000u)) {
            sm = (sm + pc);
          } else {
            sp = (sp + (-pc));
          }
        }
      } else {
        if (pc > __uint_as_float(0x00000000u)) {
          sp = (sp + pc);
        } else {
          sm = (sm + (-pc));
        }
      }
    }
    maxN = (maxN + paramsF[2]);
    if (maxN > __uint_as_float(0x3f800000u)) {
      maxN = __uint_as_float(0x3f800000u);
    }
    minN = (minN - paramsF[2]);
    if (minN < __uint_as_float(0x00000000u)) {
      minN = __uint_as_float(0x00000000u);
    }
    if (paramsF[3] > __uint_as_float(0x00000000u)) {
      float omSmooth = (__uint_as_float(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > __uint_as_float(0x3f800000u)) {
        maxN = __uint_as_float(0x3f800000u);
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < __uint_as_float(0x00000000u)) {
        minN = __uint_as_float(0x00000000u);
      }
    }
    sumPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sp;
    mSumPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sm;
    float V = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    psiMaxCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((V * rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) * (maxN - psi)) + sumPhiBD);
    psiMinCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((V * rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) * (psi - minN)) - sumPhiBD);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_phi_b_d_corr = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_phi_b_d_corr(const float* alpha, const float* phiAll, const int* owner, const int* nei, const float* alphaF, const float* alphaPhiAll, const int* params, float* phiBD, float* phiCorr) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      float ph = phiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float a = __uint_as_float(0x00000000u);
      if (ph >= __uint_as_float(0x00000000u)) {
        a = alpha[o];
      } else {
        a = alpha[n];
      }
      if (a < __uint_as_float(0x00000000u)) {
        a = __uint_as_float(0x00000000u);
      }
      if (a > __uint_as_float(0x3f800000u)) {
        a = __uint_as_float(0x3f800000u);
      }
      float pbd = (ph * a);
      phiBD[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = pbd;
      phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (alphaPhiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - pbd);
    } else {
      phiBD[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = alphaPhiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      phiCorr[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_phir = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_phir(const float* gx, const float* gy, const float* gz, const float* sfX, const float* sfY, const float* sfZ, const float* phi, const float* magSf, const int* owner, const int* nei, const int* faceParams, const float* paramsF, float* outPhir) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= faceParams[0]) {
    return;
  } else {
    if (paramsF[0] == __uint_as_float(0x00000000u)) {
      outPhir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      return;
    }
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float gox = gx[o];
    float goy = gy[o];
    float goz = gz[o];
    float gnx = gx[n];
    float gny = gy[n];
    float gnz = gz[n];
    float gfx = (__uint_as_float(0x3f000000u) * (gox + gnx));
    float gfy = (__uint_as_float(0x3f000000u) * (goy + gny));
    float gfz = (__uint_as_float(0x3f000000u) * (goz + gnz));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float dotSf = (((gfx * sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) + (gfy * sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) + (gfz * sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    float nHatf = (dotSf / (magg + paramsF[1]));
    float phic = ((paramsF[0] * abs(phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))])) / magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outPhir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phic * nHatf);
  }
}

)kernel";
const char* k_cuda_navatala_cfd_vof_rho_phi_accumulate = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_rho_phi_accumulate(const float* alphaPhi, const float* phi, float* rhoPhi, const int* counts, const float* paramsF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1]) {
    return;
  } else {
    float acc = rhoPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ap = alphaPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ph = phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    rhoPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (acc + (paramsF[2] * ((ap * paramsF[0]) + (ph * paramsF[1]))));
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_alpha_phi_all[] = {
  { "alphaPhiInt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiAll", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "alphaPhiAllOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_alpha_phi_all = {
  1,
  "navatala_cfd_vof_alpha_phi_all",
  "cuda",
  "navatala_cfd_vof_alpha_phi_all",
  "kernel:cuda:navatala_cfd_vof_alpha_phi_all",
  "abi-r1:cuda:navatala_cfd_vof_alpha_phi_all",
  "abi-r1:cuda:navatala_cfd_vof_alpha_phi_all",
  5,
  kAbiArgs_cuda_navatala_cfd_vof_alpha_phi_all
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_alpha_phi_int[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phir", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceParams", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outAlphaPhi", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_alpha_phi_int = {
  1,
  "navatala_cfd_vof_alpha_phi_int",
  "cuda",
  "navatala_cfd_vof_alpha_phi_int",
  "kernel:cuda:navatala_cfd_vof_alpha_phi_int",
  "abi-r1:cuda:navatala_cfd_vof_alpha_phi_int",
  "abi-r1:cuda:navatala_cfd_vof_alpha_phi_int",
  8,
  kAbiArgs_cuda_navatala_cfd_vof_alpha_phi_int
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_average_face_scalar_to_cell[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumMag", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outAlpha", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_average_face_scalar_to_cell = {
  1,
  "navatala_cfd_vof_average_face_scalar_to_cell",
  "cuda",
  "navatala_cfd_vof_average_face_scalar_to_cell",
  "kernel:cuda:navatala_cfd_vof_average_face_scalar_to_cell",
  "abi-r1:cuda:navatala_cfd_vof_average_face_scalar_to_cell",
  "abi-r1:cuda:navatala_cfd_vof_average_face_scalar_to_cell",
  7,
  kAbiArgs_cuda_navatala_cfd_vof_average_face_scalar_to_cell
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_interp_scalar_face_all[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcAlphaF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alphaF", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_interp_scalar_face_all = {
  1,
  "navatala_cfd_vof_interp_scalar_face_all",
  "cuda",
  "navatala_cfd_vof_interp_scalar_face_all",
  "kernel:cuda:navatala_cfd_vof_interp_scalar_face_all",
  "abi-r1:cuda:navatala_cfd_vof_interp_scalar_face_all",
  "abi-r1:cuda:navatala_cfd_vof_interp_scalar_face_all",
  7,
  kAbiArgs_cuda_navatala_cfd_vof_interp_scalar_face_all
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_alpha_face_average_internal[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_alpha_face_average_internal = {
  1,
  "navatala_cfd_vof_alpha_face_average_internal",
  "cuda",
  "navatala_cfd_vof_alpha_face_average_internal",
  "kernel:cuda:navatala_cfd_vof_alpha_face_average_internal",
  "abi-r1:cuda:navatala_cfd_vof_alpha_face_average_internal",
  "abi-r1:cuda:navatala_cfd_vof_alpha_face_average_internal",
  5,
  kAbiArgs_cuda_navatala_cfd_vof_alpha_face_average_internal
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_alpha_update[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divA", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rSubDeltaT", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaNew", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_alpha_update = {
  1,
  "navatala_cfd_vof_alpha_update",
  "cuda",
  "navatala_cfd_vof_alpha_update",
  "kernel:cuda:navatala_cfd_vof_alpha_update",
  "abi-r1:cuda:navatala_cfd_vof_alpha_update",
  "abi-r1:cuda:navatala_cfd_vof_alpha_update",
  6,
  kAbiArgs_cuda_navatala_cfd_vof_alpha_update
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_cmules_apply_correction[] = {
  { "phiCorr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "phiCorrLimited", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_cmules_apply_correction = {
  1,
  "navatala_cfd_vof_cmules_apply_correction",
  "cuda",
  "navatala_cfd_vof_cmules_apply_correction",
  "kernel:cuda:navatala_cfd_vof_cmules_apply_correction",
  "abi-r1:cuda:navatala_cfd_vof_cmules_apply_correction",
  "abi-r1:cuda:navatala_cfd_vof_cmules_apply_correction",
  5,
  kAbiArgs_cuda_navatala_cfd_vof_cmules_apply_correction
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_cmules_boundary_face_update[] = {
  { "phi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerAll", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdam", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdap", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_cmules_boundary_face_update = {
  1,
  "navatala_cfd_vof_cmules_boundary_face_update",
  "cuda",
  "navatala_cfd_vof_cmules_boundary_face_update",
  "kernel:cuda:navatala_cfd_vof_cmules_boundary_face_update",
  "abi-r1:cuda:navatala_cfd_vof_cmules_boundary_face_update",
  "abi-r1:cuda:navatala_cfd_vof_cmules_boundary_face_update",
  8,
  kAbiArgs_cuda_navatala_cfd_vof_cmules_boundary_face_update
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_cmules_correct[] = {
  { "alphaOld", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rSubDeltaT", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "su", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorrLimited", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "alphaNext", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_cmules_correct = {
  1,
  "navatala_cfd_vof_cmules_correct",
  "cuda",
  "navatala_cfd_vof_cmules_correct",
  "kernel:cuda:navatala_cfd_vof_cmules_correct",
  "abi-r1:cuda:navatala_cfd_vof_cmules_correct",
  "abi-r1:cuda:navatala_cfd_vof_cmules_correct",
  12,
  kAbiArgs_cuda_navatala_cfd_vof_cmules_correct
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_cmules_limiter_corr_prepare[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rSubDeltaT", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "su", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psiMax", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psiMin", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "psiMaxCap", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psiMinCap", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumPhip", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mSumPhim", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_cmules_limiter_corr_prepare = {
  1,
  "navatala_cfd_vof_cmules_limiter_corr_prepare",
  "cuda",
  "navatala_cfd_vof_cmules_limiter_corr_prepare",
  "kernel:cuda:navatala_cfd_vof_cmules_limiter_corr_prepare",
  "abi-r1:cuda:navatala_cfd_vof_cmules_limiter_corr_prepare",
  "abi-r1:cuda:navatala_cfd_vof_cmules_limiter_corr_prepare",
  20,
  kAbiArgs_cuda_navatala_cfd_vof_cmules_limiter_corr_prepare
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_mules_apply[] = {
  { "phiBD", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "alphaPhi", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_mules_apply = {
  1,
  "navatala_cfd_vof_mules_apply",
  "cuda",
  "navatala_cfd_vof_mules_apply",
  "kernel:cuda:navatala_cfd_vof_mules_apply",
  "abi-r1:cuda:navatala_cfd_vof_mules_apply",
  "abi-r1:cuda:navatala_cfd_vof_mules_apply",
  6,
  kAbiArgs_cuda_navatala_cfd_vof_mules_apply
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_mules_cell_lambda[] = {
  { "psiMaxCap", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psiMinCap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumPhip", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mSumPhim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumlPhip", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mSumlPhim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "lambdam", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdap", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_mules_cell_lambda = {
  1,
  "navatala_cfd_vof_mules_cell_lambda",
  "cuda",
  "navatala_cfd_vof_mules_cell_lambda",
  "kernel:cuda:navatala_cfd_vof_mules_cell_lambda",
  "abi-r1:cuda:navatala_cfd_vof_mules_cell_lambda",
  "abi-r1:cuda:navatala_cfd_vof_mules_cell_lambda",
  10,
  kAbiArgs_cuda_navatala_cfd_vof_mules_cell_lambda
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_mules_cell_sums[] = {
  { "phiCorr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "sumlPhip", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mSumlPhim", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_mules_cell_sums = {
  1,
  "navatala_cfd_vof_mules_cell_sums",
  "cuda",
  "navatala_cfd_vof_mules_cell_sums",
  "kernel:cuda:navatala_cfd_vof_mules_cell_sums",
  "abi-r1:cuda:navatala_cfd_vof_mules_cell_sums",
  "abi-r1:cuda:navatala_cfd_vof_mules_cell_sums",
  9,
  kAbiArgs_cuda_navatala_cfd_vof_mules_cell_sums
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_mules_face_update[] = {
  { "phiCorr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdam", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdap", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_mules_face_update = {
  1,
  "navatala_cfd_vof_mules_face_update",
  "cuda",
  "navatala_cfd_vof_mules_face_update",
  "kernel:cuda:navatala_cfd_vof_mules_face_update",
  "abi-r1:cuda:navatala_cfd_vof_mules_face_update",
  "abi-r1:cuda:navatala_cfd_vof_mules_face_update",
  8,
  kAbiArgs_cuda_navatala_cfd_vof_mules_face_update
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_mules_fill_lambda[] = {
  { "counts", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_mules_fill_lambda = {
  1,
  "navatala_cfd_vof_mules_fill_lambda",
  "cuda",
  "navatala_cfd_vof_mules_fill_lambda",
  "kernel:cuda:navatala_cfd_vof_mules_fill_lambda",
  "abi-r1:cuda:navatala_cfd_vof_mules_fill_lambda",
  "abi-r1:cuda:navatala_cfd_vof_mules_fill_lambda",
  3,
  kAbiArgs_cuda_navatala_cfd_vof_mules_fill_lambda
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_mules_init[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiBD", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "rSubDeltaT", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psiMaxCap", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psiMinCap", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumPhip", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mSumPhim", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_mules_init = {
  1,
  "navatala_cfd_vof_mules_init",
  "cuda",
  "navatala_cfd_vof_mules_init",
  "kernel:cuda:navatala_cfd_vof_mules_init",
  "abi-r1:cuda:navatala_cfd_vof_mules_init",
  "abi-r1:cuda:navatala_cfd_vof_mules_init",
  17,
  kAbiArgs_cuda_navatala_cfd_vof_mules_init
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_phi_b_d_corr[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiAll", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaPhiAll", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "phiBD", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_phi_b_d_corr = {
  1,
  "navatala_cfd_vof_phi_b_d_corr",
  "cuda",
  "navatala_cfd_vof_phi_b_d_corr",
  "kernel:cuda:navatala_cfd_vof_phi_b_d_corr",
  "abi-r1:cuda:navatala_cfd_vof_phi_b_d_corr",
  "abi-r1:cuda:navatala_cfd_vof_phi_b_d_corr",
  9,
  kAbiArgs_cuda_navatala_cfd_vof_phi_b_d_corr
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_phir[] = {
  { "gx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceParams", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPhir", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_phir = {
  1,
  "navatala_cfd_vof_phir",
  "cuda",
  "navatala_cfd_vof_phir",
  "kernel:cuda:navatala_cfd_vof_phir",
  "abi-r1:cuda:navatala_cfd_vof_phir",
  "abi-r1:cuda:navatala_cfd_vof_phir",
  13,
  kAbiArgs_cuda_navatala_cfd_vof_phir
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_cfd_vof_rho_phi_accumulate[] = {
  { "alphaPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoPhi", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_cfd_vof_rho_phi_accumulate = {
  1,
  "navatala_cfd_vof_rho_phi_accumulate",
  "cuda",
  "navatala_cfd_vof_rho_phi_accumulate",
  "kernel:cuda:navatala_cfd_vof_rho_phi_accumulate",
  "abi-r1:cuda:navatala_cfd_vof_rho_phi_accumulate",
  "abi-r1:cuda:navatala_cfd_vof_rho_phi_accumulate",
  5,
  kAbiArgs_cuda_navatala_cfd_vof_rho_phi_accumulate
};

bool tryGetKernelAbiManifest_cuda_cfd_vof(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_phi_all") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_alpha_phi_all;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_phi_int") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_alpha_phi_int;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_average_face_scalar_to_cell") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_average_face_scalar_to_cell;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_interp_scalar_face_all") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_interp_scalar_face_all;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_face_average_internal") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_alpha_face_average_internal;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_update") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_alpha_update;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_apply_correction") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_cmules_apply_correction;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_boundary_face_update") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_cmules_boundary_face_update;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_correct") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_cmules_correct;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_limiter_corr_prepare") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_cmules_limiter_corr_prepare;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_apply") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_mules_apply;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_cell_lambda") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_mules_cell_lambda;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_cell_sums") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_mules_cell_sums;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_face_update") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_mules_face_update;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_fill_lambda") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_mules_fill_lambda;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_init") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_mules_init;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_phi_b_d_corr") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_phi_b_d_corr;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_phir") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_phir;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_rho_phi_accumulate") {
    out = &kAbiManifest_cuda_navatala_cfd_vof_rho_phi_accumulate;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_cfd_vof(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_phi_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_alpha_phi_all";
    std::string_view sv(k_cuda_navatala_cfd_vof_alpha_phi_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_phi_int") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_alpha_phi_int";
    std::string_view sv(k_cuda_navatala_cfd_vof_alpha_phi_int);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_average_face_scalar_to_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_average_face_scalar_to_cell";
    std::string_view sv(k_cuda_navatala_cfd_vof_average_face_scalar_to_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_interp_scalar_face_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_interp_scalar_face_all";
    std::string_view sv(k_cuda_navatala_cfd_vof_interp_scalar_face_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_face_average_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_alpha_face_average_internal";
    std::string_view sv(k_cuda_navatala_cfd_vof_alpha_face_average_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_alpha_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_alpha_update";
    std::string_view sv(k_cuda_navatala_cfd_vof_alpha_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_apply_correction") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_cmules_apply_correction";
    std::string_view sv(k_cuda_navatala_cfd_vof_cmules_apply_correction);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_boundary_face_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_cmules_boundary_face_update";
    std::string_view sv(k_cuda_navatala_cfd_vof_cmules_boundary_face_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_cmules_correct";
    std::string_view sv(k_cuda_navatala_cfd_vof_cmules_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_cmules_limiter_corr_prepare") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_cmules_limiter_corr_prepare";
    std::string_view sv(k_cuda_navatala_cfd_vof_cmules_limiter_corr_prepare);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_apply") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_mules_apply";
    std::string_view sv(k_cuda_navatala_cfd_vof_mules_apply);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_cell_lambda") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_mules_cell_lambda";
    std::string_view sv(k_cuda_navatala_cfd_vof_mules_cell_lambda);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_cell_sums") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_mules_cell_sums";
    std::string_view sv(k_cuda_navatala_cfd_vof_mules_cell_sums);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_face_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_mules_face_update";
    std::string_view sv(k_cuda_navatala_cfd_vof_mules_face_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_fill_lambda") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_mules_fill_lambda";
    std::string_view sv(k_cuda_navatala_cfd_vof_mules_fill_lambda);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_mules_init") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_mules_init";
    std::string_view sv(k_cuda_navatala_cfd_vof_mules_init);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_phi_b_d_corr") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_phi_b_d_corr";
    std::string_view sv(k_cuda_navatala_cfd_vof_phi_b_d_corr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_phir") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_phir";
    std::string_view sv(k_cuda_navatala_cfd_vof_phir);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_cfd_vof_rho_phi_accumulate") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_cfd_vof_rho_phi_accumulate";
    std::string_view sv(k_cuda_navatala_cfd_vof_rho_phi_accumulate);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

