// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_cfd_vof` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_cfd_vof_alpha_phi_all = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_alpha_phi_all(device const float* alphaPhiInt [[buffer(0)]], device const float* phiAll [[buffer(1)]], device const float* alphaF [[buffer(2)]], device const int* params [[buffer(3)]], device float* alphaPhiAllOut [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    if (((int)(int(__gid.x))) < params[1]) {
      alphaPhiAllOut[((int)(int(__gid.x)))] = alphaPhiInt[((int)(int(__gid.x)))];
    } else {
      float a = alphaF[((int)(int(__gid.x)))];
      if (a < as_type<float>(0x00000000u)) {
        a = as_type<float>(0x00000000u);
      }
      if (a > as_type<float>(0x3f800000u)) {
        a = as_type<float>(0x3f800000u);
      }
      alphaPhiAllOut[((int)(int(__gid.x)))] = (phiAll[((int)(int(__gid.x)))] * a);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_alpha_phi_int = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_alpha_phi_int(device const float* alpha [[buffer(0)]], device const float* phi [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device const float* phir [[buffer(4)]], device const float* alphaF [[buffer(5)]], device const int* faceParams [[buffer(6)]], device float* outAlphaPhi [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)(int(__gid.x)))];
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float a = as_type<float>(0x00000000u);
    if (ph >= as_type<float>(0x00000000u)) {
      a = alpha[o];
    } else {
      a = alpha[n];
    }
    if (a < as_type<float>(0x00000000u)) {
      a = as_type<float>(0x00000000u);
    }
    if (a > as_type<float>(0x3f800000u)) {
      a = as_type<float>(0x3f800000u);
    }
    float ac = alphaF[((int)(int(__gid.x)))];
    if (ac < as_type<float>(0x00000000u)) {
      ac = as_type<float>(0x00000000u);
    }
    if (ac > as_type<float>(0x3f800000u)) {
      ac = as_type<float>(0x3f800000u);
    }
    float oneMinus = (as_type<float>(0x3f800000u) - ac);
    float comp = (ac * oneMinus);
    outAlphaPhi[((int)(int(__gid.x)))] = ((ph * a) + (phir[((int)(int(__gid.x)))] * comp));
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_average_face_scalar_to_cell = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_average_face_scalar_to_cell(device const float* alphaF [[buffer(0)]], device const float* magSf [[buffer(1)]], device const uint* offsets [[buffer(2)]], device const uint* faceIdx [[buffer(3)]], device const float* sumMag [[buffer(4)]], device const uint* counts [[buffer(5)]], device float* outAlpha [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = as_type<float>(0x00000000u);
    uint beg = offsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      s = (s + (magSf[((int)(f))] * alphaF[((int)(f))]));
    }
    float d = sumMag[int(__gid.x)];
    if (d < as_type<float>(0x1e3ce508u)) {
      outAlpha[int(__gid.x)] = (s / as_type<float>(0x1e3ce508u));
    } else {
      outAlpha[int(__gid.x)] = (s / d);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_interp_scalar_face_all = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_interp_scalar_face_all(device const float* alpha [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* nei [[buffer(2)]], device const float* w [[buffer(3)]], device const float* bcAlphaF [[buffer(4)]], device const uint* params [[buffer(5)]], device float* alphaF [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[0]))) {
    return;
  } else {
    if (int(__gid.x) < ((int)(params[1]))) {
      int o = owner[int(__gid.x)];
      int n = nei[int(__gid.x)];
      float wf = w[int(__gid.x)];
      float wc = (wf * ((float)((wf > as_type<float>(0x00000000u)))));
      float a = ((wc * alpha[o]) + ((as_type<float>(0x3f800000u) - wc) * alpha[n]));
      float ac = (a * ((float)((a > as_type<float>(0x00000000u)))));
      if (ac > as_type<float>(0x3f800000u)) {
        alphaF[int(__gid.x)] = as_type<float>(0x3f800000u);
      } else {
        alphaF[int(__gid.x)] = ac;
      }
    } else {
      float a = bcAlphaF[int(__gid.x)];
      float ac = (a * ((float)((a > as_type<float>(0x00000000u)))));
      if (ac > as_type<float>(0x3f800000u)) {
        alphaF[int(__gid.x)] = as_type<float>(0x3f800000u);
      } else {
        alphaF[int(__gid.x)] = ac;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_alpha_update = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_alpha_update(device const float* alpha [[buffer(0)]], device const float* divA [[buffer(1)]], device const float* paramsF [[buffer(2)]], device const int* counts [[buffer(3)]], device const float* rSubDeltaT [[buffer(4)]], device float* alphaNew [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float a = (alpha[((int)(int(__gid.x)))] - (divA[((int)(int(__gid.x)))] / rSubDeltaT[((int)(int(__gid.x)))]));
    if (a < as_type<float>(0x00000000u)) {
      a = as_type<float>(0x00000000u);
    }
    if (a > as_type<float>(0x3f800000u)) {
      a = as_type<float>(0x3f800000u);
    }
    alphaNew[((int)(int(__gid.x)))] = a;
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_mules_apply = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_mules_apply(device const float* phiBD [[buffer(0)]], device const float* phiCorr [[buffer(1)]], device const float* lambda [[buffer(2)]], device const int* counts [[buffer(3)]], device const float* paramsF [[buffer(4)]], device float* alphaPhi [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    alphaPhi[((int)(int(__gid.x)))] = (phiBD[((int)(int(__gid.x)))] + (lambda[((int)(int(__gid.x)))] * phiCorr[((int)(int(__gid.x)))]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_mules_cell_lambda = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_mules_cell_lambda(device const float* psiMaxCap [[buffer(0)]], device const float* psiMinCap [[buffer(1)]], device const float* sumPhip [[buffer(2)]], device const float* mSumPhim [[buffer(3)]], device const float* sumlPhip [[buffer(4)]], device const float* mSumlPhim [[buffer(5)]], device const int* counts [[buffer(6)]], device const float* paramsF [[buffer(7)]], device float* lambdam [[buffer(8)]], device float* lambdap [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float ld = ((sumlPhip[((int)(int(__gid.x)))] + psiMaxCap[((int)(int(__gid.x)))]) / (mSumPhim[((int)(int(__gid.x)))] + paramsF[1]));
    float lp = ((mSumlPhim[((int)(int(__gid.x)))] + psiMinCap[((int)(int(__gid.x)))]) / (sumPhip[((int)(int(__gid.x)))] + paramsF[1]));
    if (ld < as_type<float>(0x00000000u)) {
      ld = as_type<float>(0x00000000u);
    }
    if (ld > as_type<float>(0x3f800000u)) {
      ld = as_type<float>(0x3f800000u);
    }
    if (lp < as_type<float>(0x00000000u)) {
      lp = as_type<float>(0x00000000u);
    }
    if (lp > as_type<float>(0x3f800000u)) {
      lp = as_type<float>(0x3f800000u);
    }
    lambdam[((int)(int(__gid.x)))] = ld;
    lambdap[((int)(int(__gid.x)))] = lp;
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_mules_cell_sums = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_mules_cell_sums(device const float* phiCorr [[buffer(0)]], device const float* lambda [[buffer(1)]], device const int* offsets [[buffer(2)]], device const int* faceIdx [[buffer(3)]], device const float* sign [[buffer(4)]], device const int* counts [[buffer(5)]], device const float* paramsF [[buffer(6)]], device float* sumlPhip [[buffer(7)]], device float* mSumlPhim [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sp = as_type<float>(0x00000000u);
    float sm = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float lpc = (lambda[f] * phiCorr[f]);
      float s = sign[k];
      if (f < counts[2]) {
        if (s > as_type<float>(0x00000000u)) {
          if (lpc > as_type<float>(0x00000000u)) {
            sp = (sp + lpc);
          } else {
            sm = (sm + (-lpc));
          }
        } else {
          if (lpc > as_type<float>(0x00000000u)) {
            sm = (sm + lpc);
          } else {
            sp = (sp + (-lpc));
          }
        }
      } else {
        if (lpc > as_type<float>(0x00000000u)) {
          sp = (sp + lpc);
        } else {
          sm = (sm + (-lpc));
        }
      }
    }
    sumlPhip[((int)(int(__gid.x)))] = sp;
    mSumlPhim[((int)(int(__gid.x)))] = sm;
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_mules_face_update = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_mules_face_update(device const float* phiCorr [[buffer(0)]], device float* lambda [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device const float* lambdam [[buffer(4)]], device const float* lambdap [[buffer(5)]], device const int* counts [[buffer(6)]], device const float* paramsF [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[2]) {
    return;
  } else {
    float pc = phiCorr[((int)(int(__gid.x)))];
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float a0 = lambdam[o];
    float b0 = lambdap[n];
    float lim = a0;
    if (lim > b0) {
      lim = b0;
    }
    if (pc > as_type<float>(0x00000000u)) {
      float a1 = lambdap[o];
      float b1 = lambdam[n];
      float lim1 = a1;
      if (lim1 > b1) {
        lim1 = b1;
      }
      lim = lim1;
    }
    float cur = lambda[((int)(int(__gid.x)))];
    float _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[((int)(int(__gid.x)))] = _out;
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_mules_fill_lambda = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_mules_fill_lambda(device const int* counts [[buffer(0)]], device const float* paramsF [[buffer(1)]], device float* lambda [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    lambda[((int)(int(__gid.x)))] = as_type<float>(0x3f800000u);
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_mules_init = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_mules_init(device const float* alpha [[buffer(0)]], device const float* alphaF [[buffer(1)]], device const float* phiBD [[buffer(2)]], device const float* phiCorr [[buffer(3)]], device const int* offsets [[buffer(4)]], device const int* faceIdx [[buffer(5)]], device const float* sign [[buffer(6)]], device const int* owner [[buffer(7)]], device const int* nei [[buffer(8)]], device const float* vol [[buffer(9)]], device const int* counts [[buffer(10)]], device const float* paramsF [[buffer(11)]], device const float* rSubDeltaT [[buffer(12)]], device float* psiMaxCap [[buffer(13)]], device float* psiMinCap [[buffer(14)]], device float* sumPhip [[buffer(15)]], device float* mSumPhim [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)(int(__gid.x)))];
    if (psi < as_type<float>(0x00000000u)) {
      psi = as_type<float>(0x00000000u);
    }
    if (psi > as_type<float>(0x3f800000u)) {
      psi = as_type<float>(0x3f800000u);
    }
    float maxN = as_type<float>(0x00000000u);
    float minN = as_type<float>(0x3f800000u);
    float sumPhiBD = as_type<float>(0x00000000u);
    float sp = as_type<float>(0x00000000u);
    float sm = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = as_type<float>(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > as_type<float>(0x00000000u)) {
          nbr = nei[f];
        }
        v = alpha[nbr];
      } else {
        v = alphaF[f];
      }
      if (v < as_type<float>(0x00000000u)) {
        v = as_type<float>(0x00000000u);
      }
      if (v > as_type<float>(0x3f800000u)) {
        v = as_type<float>(0x3f800000u);
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
        if (s > as_type<float>(0x00000000u)) {
          if (pc > as_type<float>(0x00000000u)) {
            sp = (sp + pc);
          } else {
            sm = (sm + (-pc));
          }
        } else {
          if (pc > as_type<float>(0x00000000u)) {
            sm = (sm + pc);
          } else {
            sp = (sp + (-pc));
          }
        }
      } else {
        if (pc > as_type<float>(0x00000000u)) {
          sp = (sp + pc);
        } else {
          sm = (sm + (-pc));
        }
      }
    }
    maxN = (maxN + paramsF[2]);
    if (maxN > as_type<float>(0x3f800000u)) {
      maxN = as_type<float>(0x3f800000u);
    }
    minN = (minN - paramsF[2]);
    if (minN < as_type<float>(0x00000000u)) {
      minN = as_type<float>(0x00000000u);
    }
    if (paramsF[3] > as_type<float>(0x00000000u)) {
      float omSmooth = (as_type<float>(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > as_type<float>(0x3f800000u)) {
        maxN = as_type<float>(0x3f800000u);
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < as_type<float>(0x00000000u)) {
        minN = as_type<float>(0x00000000u);
      }
    }
    sumPhip[((int)(int(__gid.x)))] = sp;
    mSumPhim[((int)(int(__gid.x)))] = sm;
    float V = vol[((int)(int(__gid.x)))];
    psiMaxCap[((int)(int(__gid.x)))] = (((V * rSubDeltaT[((int)(int(__gid.x)))]) * (maxN - psi)) + sumPhiBD);
    psiMinCap[((int)(int(__gid.x)))] = (((V * rSubDeltaT[((int)(int(__gid.x)))]) * (psi - minN)) - sumPhiBD);
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_phi_b_d_corr = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_phi_b_d_corr(device const float* alpha [[buffer(0)]], device const float* phiAll [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device const float* alphaF [[buffer(4)]], device const float* alphaPhiAll [[buffer(5)]], device const int* params [[buffer(6)]], device float* phiBD [[buffer(7)]], device float* phiCorr [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    if (((int)(int(__gid.x))) < params[1]) {
      float ph = phiAll[((int)(int(__gid.x)))];
      int o = owner[((int)(int(__gid.x)))];
      int n = nei[((int)(int(__gid.x)))];
      float a = as_type<float>(0x00000000u);
      if (ph >= as_type<float>(0x00000000u)) {
        a = alpha[o];
      } else {
        a = alpha[n];
      }
      if (a < as_type<float>(0x00000000u)) {
        a = as_type<float>(0x00000000u);
      }
      if (a > as_type<float>(0x3f800000u)) {
        a = as_type<float>(0x3f800000u);
      }
      float pbd = (ph * a);
      phiBD[((int)(int(__gid.x)))] = pbd;
      phiCorr[((int)(int(__gid.x)))] = (alphaPhiAll[((int)(int(__gid.x)))] - pbd);
    } else {
      phiBD[((int)(int(__gid.x)))] = alphaPhiAll[((int)(int(__gid.x)))];
      phiCorr[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_phir = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_phir(device const float* gx [[buffer(0)]], device const float* gy [[buffer(1)]], device const float* gz [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* phi [[buffer(6)]], device const float* magSf [[buffer(7)]], device const int* owner [[buffer(8)]], device const int* nei [[buffer(9)]], device const int* faceParams [[buffer(10)]], device const float* paramsF [[buffer(11)]], device float* outPhir [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= faceParams[0]) {
    return;
  } else {
    if (paramsF[0] == as_type<float>(0x00000000u)) {
      outPhir[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
      return;
    }
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float gox = gx[o];
    float goy = gy[o];
    float goz = gz[o];
    float gnx = gx[n];
    float gny = gy[n];
    float gnz = gz[n];
    float gfx = (as_type<float>(0x3f000000u) * (gox + gnx));
    float gfy = (as_type<float>(0x3f000000u) * (goy + gny));
    float gfz = (as_type<float>(0x3f000000u) * (goz + gnz));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float dotSf = (((gfx * sfX[((int)(int(__gid.x)))]) + (gfy * sfY[((int)(int(__gid.x)))])) + (gfz * sfZ[((int)(int(__gid.x)))]));
    float nHatf = (dotSf / (magg + paramsF[1]));
    float phic = ((paramsF[0] * abs(phi[((int)(int(__gid.x)))])) / magSf[((int)(int(__gid.x)))]);
    outPhir[((int)(int(__gid.x)))] = (phic * nHatf);
  }
}

)kernel";
const char* k_metal_navatala_cfd_vof_rho_phi_accumulate = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_rho_phi_accumulate(device const float* alphaPhi [[buffer(0)]], device const float* phi [[buffer(1)]], device float* rhoPhi [[buffer(2)]], device const int* counts [[buffer(3)]], device const float* paramsF [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    float acc = rhoPhi[((int)(int(__gid.x)))];
    float ap = alphaPhi[((int)(int(__gid.x)))];
    float ph = phi[((int)(int(__gid.x)))];
    rhoPhi[((int)(int(__gid.x)))] = (acc + (paramsF[2] * ((ap * paramsF[0]) + (ph * paramsF[1]))));
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_alpha_phi_all[] = {
  { "alphaPhiInt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiAll", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "alphaPhiAllOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_alpha_phi_all = {
  1,
  "navatala_cfd_vof_alpha_phi_all",
  "metal",
  "navatala_cfd_vof_alpha_phi_all",
  "kernel:metal:navatala_cfd_vof_alpha_phi_all",
  "abi-r1:metal:navatala_cfd_vof_alpha_phi_all",
  "abi-r1:metal:navatala_cfd_vof_alpha_phi_all",
  5,
  kAbiArgs_metal_navatala_cfd_vof_alpha_phi_all
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_alpha_phi_int[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phir", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceParams", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outAlphaPhi", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_alpha_phi_int = {
  1,
  "navatala_cfd_vof_alpha_phi_int",
  "metal",
  "navatala_cfd_vof_alpha_phi_int",
  "kernel:metal:navatala_cfd_vof_alpha_phi_int",
  "abi-r1:metal:navatala_cfd_vof_alpha_phi_int",
  "abi-r1:metal:navatala_cfd_vof_alpha_phi_int",
  8,
  kAbiArgs_metal_navatala_cfd_vof_alpha_phi_int
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_average_face_scalar_to_cell[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumMag", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outAlpha", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_average_face_scalar_to_cell = {
  1,
  "navatala_cfd_vof_average_face_scalar_to_cell",
  "metal",
  "navatala_cfd_vof_average_face_scalar_to_cell",
  "kernel:metal:navatala_cfd_vof_average_face_scalar_to_cell",
  "abi-r1:metal:navatala_cfd_vof_average_face_scalar_to_cell",
  "abi-r1:metal:navatala_cfd_vof_average_face_scalar_to_cell",
  7,
  kAbiArgs_metal_navatala_cfd_vof_average_face_scalar_to_cell
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_interp_scalar_face_all[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcAlphaF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alphaF", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_interp_scalar_face_all = {
  1,
  "navatala_cfd_vof_interp_scalar_face_all",
  "metal",
  "navatala_cfd_vof_interp_scalar_face_all",
  "kernel:metal:navatala_cfd_vof_interp_scalar_face_all",
  "abi-r1:metal:navatala_cfd_vof_interp_scalar_face_all",
  "abi-r1:metal:navatala_cfd_vof_interp_scalar_face_all",
  7,
  kAbiArgs_metal_navatala_cfd_vof_interp_scalar_face_all
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_alpha_update[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divA", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rSubDeltaT", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaNew", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_alpha_update = {
  1,
  "navatala_cfd_vof_alpha_update",
  "metal",
  "navatala_cfd_vof_alpha_update",
  "kernel:metal:navatala_cfd_vof_alpha_update",
  "abi-r1:metal:navatala_cfd_vof_alpha_update",
  "abi-r1:metal:navatala_cfd_vof_alpha_update",
  6,
  kAbiArgs_metal_navatala_cfd_vof_alpha_update
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_mules_apply[] = {
  { "phiBD", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "alphaPhi", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_mules_apply = {
  1,
  "navatala_cfd_vof_mules_apply",
  "metal",
  "navatala_cfd_vof_mules_apply",
  "kernel:metal:navatala_cfd_vof_mules_apply",
  "abi-r1:metal:navatala_cfd_vof_mules_apply",
  "abi-r1:metal:navatala_cfd_vof_mules_apply",
  6,
  kAbiArgs_metal_navatala_cfd_vof_mules_apply
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_mules_cell_lambda[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_mules_cell_lambda = {
  1,
  "navatala_cfd_vof_mules_cell_lambda",
  "metal",
  "navatala_cfd_vof_mules_cell_lambda",
  "kernel:metal:navatala_cfd_vof_mules_cell_lambda",
  "abi-r1:metal:navatala_cfd_vof_mules_cell_lambda",
  "abi-r1:metal:navatala_cfd_vof_mules_cell_lambda",
  10,
  kAbiArgs_metal_navatala_cfd_vof_mules_cell_lambda
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_mules_cell_sums[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_mules_cell_sums = {
  1,
  "navatala_cfd_vof_mules_cell_sums",
  "metal",
  "navatala_cfd_vof_mules_cell_sums",
  "kernel:metal:navatala_cfd_vof_mules_cell_sums",
  "abi-r1:metal:navatala_cfd_vof_mules_cell_sums",
  "abi-r1:metal:navatala_cfd_vof_mules_cell_sums",
  9,
  kAbiArgs_metal_navatala_cfd_vof_mules_cell_sums
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_mules_face_update[] = {
  { "phiCorr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdam", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdap", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_mules_face_update = {
  1,
  "navatala_cfd_vof_mules_face_update",
  "metal",
  "navatala_cfd_vof_mules_face_update",
  "kernel:metal:navatala_cfd_vof_mules_face_update",
  "abi-r1:metal:navatala_cfd_vof_mules_face_update",
  "abi-r1:metal:navatala_cfd_vof_mules_face_update",
  8,
  kAbiArgs_metal_navatala_cfd_vof_mules_face_update
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_mules_fill_lambda[] = {
  { "counts", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_mules_fill_lambda = {
  1,
  "navatala_cfd_vof_mules_fill_lambda",
  "metal",
  "navatala_cfd_vof_mules_fill_lambda",
  "kernel:metal:navatala_cfd_vof_mules_fill_lambda",
  "abi-r1:metal:navatala_cfd_vof_mules_fill_lambda",
  "abi-r1:metal:navatala_cfd_vof_mules_fill_lambda",
  3,
  kAbiArgs_metal_navatala_cfd_vof_mules_fill_lambda
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_mules_init[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_mules_init = {
  1,
  "navatala_cfd_vof_mules_init",
  "metal",
  "navatala_cfd_vof_mules_init",
  "kernel:metal:navatala_cfd_vof_mules_init",
  "abi-r1:metal:navatala_cfd_vof_mules_init",
  "abi-r1:metal:navatala_cfd_vof_mules_init",
  17,
  kAbiArgs_metal_navatala_cfd_vof_mules_init
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_phi_b_d_corr[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_phi_b_d_corr = {
  1,
  "navatala_cfd_vof_phi_b_d_corr",
  "metal",
  "navatala_cfd_vof_phi_b_d_corr",
  "kernel:metal:navatala_cfd_vof_phi_b_d_corr",
  "abi-r1:metal:navatala_cfd_vof_phi_b_d_corr",
  "abi-r1:metal:navatala_cfd_vof_phi_b_d_corr",
  9,
  kAbiArgs_metal_navatala_cfd_vof_phi_b_d_corr
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_phir[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_phir = {
  1,
  "navatala_cfd_vof_phir",
  "metal",
  "navatala_cfd_vof_phir",
  "kernel:metal:navatala_cfd_vof_phir",
  "abi-r1:metal:navatala_cfd_vof_phir",
  "abi-r1:metal:navatala_cfd_vof_phir",
  13,
  kAbiArgs_metal_navatala_cfd_vof_phir
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_vof_rho_phi_accumulate[] = {
  { "alphaPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoPhi", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_vof_rho_phi_accumulate = {
  1,
  "navatala_cfd_vof_rho_phi_accumulate",
  "metal",
  "navatala_cfd_vof_rho_phi_accumulate",
  "kernel:metal:navatala_cfd_vof_rho_phi_accumulate",
  "abi-r1:metal:navatala_cfd_vof_rho_phi_accumulate",
  "abi-r1:metal:navatala_cfd_vof_rho_phi_accumulate",
  5,
  kAbiArgs_metal_navatala_cfd_vof_rho_phi_accumulate
};

bool tryGetKernelAbiManifest_metal_cfd_vof(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_cfd_vof_alpha_phi_all") {
    out = &kAbiManifest_metal_navatala_cfd_vof_alpha_phi_all;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_alpha_phi_int") {
    out = &kAbiManifest_metal_navatala_cfd_vof_alpha_phi_int;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_average_face_scalar_to_cell") {
    out = &kAbiManifest_metal_navatala_cfd_vof_average_face_scalar_to_cell;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_interp_scalar_face_all") {
    out = &kAbiManifest_metal_navatala_cfd_vof_interp_scalar_face_all;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_alpha_update") {
    out = &kAbiManifest_metal_navatala_cfd_vof_alpha_update;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_apply") {
    out = &kAbiManifest_metal_navatala_cfd_vof_mules_apply;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_cell_lambda") {
    out = &kAbiManifest_metal_navatala_cfd_vof_mules_cell_lambda;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_cell_sums") {
    out = &kAbiManifest_metal_navatala_cfd_vof_mules_cell_sums;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_face_update") {
    out = &kAbiManifest_metal_navatala_cfd_vof_mules_face_update;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_fill_lambda") {
    out = &kAbiManifest_metal_navatala_cfd_vof_mules_fill_lambda;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_init") {
    out = &kAbiManifest_metal_navatala_cfd_vof_mules_init;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_phi_b_d_corr") {
    out = &kAbiManifest_metal_navatala_cfd_vof_phi_b_d_corr;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_phir") {
    out = &kAbiManifest_metal_navatala_cfd_vof_phir;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_rho_phi_accumulate") {
    out = &kAbiManifest_metal_navatala_cfd_vof_rho_phi_accumulate;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_cfd_vof(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_cfd_vof_alpha_phi_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_alpha_phi_all";
    std::string_view sv(k_metal_navatala_cfd_vof_alpha_phi_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_alpha_phi_int") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_alpha_phi_int";
    std::string_view sv(k_metal_navatala_cfd_vof_alpha_phi_int);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_average_face_scalar_to_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_average_face_scalar_to_cell";
    std::string_view sv(k_metal_navatala_cfd_vof_average_face_scalar_to_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_interp_scalar_face_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_interp_scalar_face_all";
    std::string_view sv(k_metal_navatala_cfd_vof_interp_scalar_face_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_alpha_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_alpha_update";
    std::string_view sv(k_metal_navatala_cfd_vof_alpha_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_apply") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_mules_apply";
    std::string_view sv(k_metal_navatala_cfd_vof_mules_apply);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_cell_lambda") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_mules_cell_lambda";
    std::string_view sv(k_metal_navatala_cfd_vof_mules_cell_lambda);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_cell_sums") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_mules_cell_sums";
    std::string_view sv(k_metal_navatala_cfd_vof_mules_cell_sums);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_face_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_mules_face_update";
    std::string_view sv(k_metal_navatala_cfd_vof_mules_face_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_fill_lambda") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_mules_fill_lambda";
    std::string_view sv(k_metal_navatala_cfd_vof_mules_fill_lambda);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_mules_init") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_mules_init";
    std::string_view sv(k_metal_navatala_cfd_vof_mules_init);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_phi_b_d_corr") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_phi_b_d_corr";
    std::string_view sv(k_metal_navatala_cfd_vof_phi_b_d_corr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_phir") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_phir";
    std::string_view sv(k_metal_navatala_cfd_vof_phir);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_vof_rho_phi_accumulate") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_vof_rho_phi_accumulate";
    std::string_view sv(k_metal_navatala_cfd_vof_rho_phi_accumulate);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

