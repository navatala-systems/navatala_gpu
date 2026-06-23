// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_cfd_vof` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_cfd_vof_alpha_phi_all = R"kernel(
__kernel void navatala_cfd_vof_alpha_phi_all(__global const float* alphaPhiInt, __global const float* phiAll, __global const float* alphaF, __global const int* params, __global float* alphaPhiAllOut) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    if (((int)((int)(get_global_id(0)))) < params[1]) {
      alphaPhiAllOut[((int)((int)(get_global_id(0))))] = alphaPhiInt[((int)((int)(get_global_id(0))))];
    } else {
      float a = alphaF[((int)((int)(get_global_id(0))))];
      if (a < as_float(0x00000000u)) {
        a = as_float(0x00000000u);
      }
      if (a > as_float(0x3f800000u)) {
        a = as_float(0x3f800000u);
      }
      alphaPhiAllOut[((int)((int)(get_global_id(0))))] = (phiAll[((int)((int)(get_global_id(0))))] * a);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_alpha_phi_int = R"kernel(
__kernel void navatala_cfd_vof_alpha_phi_int(__global const float* alpha, __global const float* phi, __global const int* owner, __global const int* nei, __global const float* phir, __global const float* alphaF, __global const int* faceParams, __global float* outAlphaPhi) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)((int)(get_global_id(0))))];
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float a = as_float(0x00000000u);
    if (ph >= as_float(0x00000000u)) {
      a = alpha[o];
    } else {
      a = alpha[n];
    }
    if (a < as_float(0x00000000u)) {
      a = as_float(0x00000000u);
    }
    if (a > as_float(0x3f800000u)) {
      a = as_float(0x3f800000u);
    }
    float ac = alphaF[((int)((int)(get_global_id(0))))];
    if (ac < as_float(0x00000000u)) {
      ac = as_float(0x00000000u);
    }
    if (ac > as_float(0x3f800000u)) {
      ac = as_float(0x3f800000u);
    }
    float oneMinus = (as_float(0x3f800000u) - ac);
    float comp = (ac * oneMinus);
    outAlphaPhi[((int)((int)(get_global_id(0))))] = ((ph * a) + (phir[((int)((int)(get_global_id(0))))] * comp));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_average_face_scalar_to_cell = R"kernel(
__kernel void navatala_cfd_vof_average_face_scalar_to_cell(__global const float* alphaF, __global const float* magSf, __global const uint* offsets, __global const uint* faceIdx, __global const float* sumMag, __global const uint* counts, __global float* outAlpha) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = as_float(0x00000000u);
    uint beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      s = (s + (magSf[((int)(f))] * alphaF[((int)(f))]));
    }
    float d = sumMag[(int)(get_global_id(0))];
    if (d < as_float(0x1e3ce508u)) {
      outAlpha[(int)(get_global_id(0))] = (s / as_float(0x1e3ce508u));
    } else {
      outAlpha[(int)(get_global_id(0))] = (s / d);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_interp_scalar_face_all = R"kernel(
__kernel void navatala_cfd_vof_interp_scalar_face_all(__global const float* alpha, __global const int* owner, __global const int* nei, __global const float* w, __global const float* bcAlphaF, __global const uint* params, __global float* alphaF) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(get_global_id(0)) < ((int)(params[1]))) {
      int o = owner[(int)(get_global_id(0))];
      int n = nei[(int)(get_global_id(0))];
      float wf = w[(int)(get_global_id(0))];
      float wc = (wf * ((float)((wf > as_float(0x00000000u)))));
      float a = ((wc * alpha[o]) + ((as_float(0x3f800000u) - wc) * alpha[n]));
      float ac = (a * ((float)((a > as_float(0x00000000u)))));
      if (ac > as_float(0x3f800000u)) {
        alphaF[(int)(get_global_id(0))] = as_float(0x3f800000u);
      } else {
        alphaF[(int)(get_global_id(0))] = ac;
      }
    } else {
      float a = bcAlphaF[(int)(get_global_id(0))];
      float ac = (a * ((float)((a > as_float(0x00000000u)))));
      if (ac > as_float(0x3f800000u)) {
        alphaF[(int)(get_global_id(0))] = as_float(0x3f800000u);
      } else {
        alphaF[(int)(get_global_id(0))] = ac;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_alpha_update = R"kernel(
__kernel void navatala_cfd_vof_alpha_update(__global const float* alpha, __global const float* divA, __global const float* paramsF, __global const int* counts, __global const float* rSubDeltaT, __global float* alphaNew) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float a = (alpha[((int)((int)(get_global_id(0))))] - (divA[((int)((int)(get_global_id(0))))] / rSubDeltaT[((int)((int)(get_global_id(0))))]));
    if (a < as_float(0x00000000u)) {
      a = as_float(0x00000000u);
    }
    if (a > as_float(0x3f800000u)) {
      a = as_float(0x3f800000u);
    }
    alphaNew[((int)((int)(get_global_id(0))))] = a;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_mules_apply = R"kernel(
__kernel void navatala_cfd_vof_mules_apply(__global const float* phiBD, __global const float* phiCorr, __global const float* lambda, __global const int* counts, __global const float* paramsF, __global float* alphaPhi) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    alphaPhi[((int)((int)(get_global_id(0))))] = (phiBD[((int)((int)(get_global_id(0))))] + (lambda[((int)((int)(get_global_id(0))))] * phiCorr[((int)((int)(get_global_id(0))))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_mules_cell_lambda = R"kernel(
__kernel void navatala_cfd_vof_mules_cell_lambda(__global const float* psiMaxCap, __global const float* psiMinCap, __global const float* sumPhip, __global const float* mSumPhim, __global const float* sumlPhip, __global const float* mSumlPhim, __global const int* counts, __global const float* paramsF, __global float* lambdam, __global float* lambdap) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float ld = ((sumlPhip[((int)((int)(get_global_id(0))))] + psiMaxCap[((int)((int)(get_global_id(0))))]) / (mSumPhim[((int)((int)(get_global_id(0))))] + paramsF[1]));
    float lp = ((mSumlPhim[((int)((int)(get_global_id(0))))] + psiMinCap[((int)((int)(get_global_id(0))))]) / (sumPhip[((int)((int)(get_global_id(0))))] + paramsF[1]));
    if (ld < as_float(0x00000000u)) {
      ld = as_float(0x00000000u);
    }
    if (ld > as_float(0x3f800000u)) {
      ld = as_float(0x3f800000u);
    }
    if (lp < as_float(0x00000000u)) {
      lp = as_float(0x00000000u);
    }
    if (lp > as_float(0x3f800000u)) {
      lp = as_float(0x3f800000u);
    }
    lambdam[((int)((int)(get_global_id(0))))] = ld;
    lambdap[((int)((int)(get_global_id(0))))] = lp;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_mules_cell_sums = R"kernel(
__kernel void navatala_cfd_vof_mules_cell_sums(__global const float* phiCorr, __global const float* lambda, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const int* counts, __global const float* paramsF, __global float* sumlPhip, __global float* mSumlPhim) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sp = as_float(0x00000000u);
    float sm = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float lpc = (lambda[f] * phiCorr[f]);
      float s = sign[k];
      if (f < counts[2]) {
        if (s > as_float(0x00000000u)) {
          if (lpc > as_float(0x00000000u)) {
            sp = (sp + lpc);
          } else {
            sm = (sm + (-lpc));
          }
        } else {
          if (lpc > as_float(0x00000000u)) {
            sm = (sm + lpc);
          } else {
            sp = (sp + (-lpc));
          }
        }
      } else {
        if (lpc > as_float(0x00000000u)) {
          sp = (sp + lpc);
        } else {
          sm = (sm + (-lpc));
        }
      }
    }
    sumlPhip[((int)((int)(get_global_id(0))))] = sp;
    mSumlPhim[((int)((int)(get_global_id(0))))] = sm;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_mules_face_update = R"kernel(
__kernel void navatala_cfd_vof_mules_face_update(__global const float* phiCorr, __global float* lambda, __global const int* owner, __global const int* nei, __global const float* lambdam, __global const float* lambdap, __global const int* counts, __global const float* paramsF) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[2]) {
    return;
  } else {
    float pc = phiCorr[((int)((int)(get_global_id(0))))];
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float a0 = lambdam[o];
    float b0 = lambdap[n];
    float lim = a0;
    if (lim > b0) {
      lim = b0;
    }
    if (pc > as_float(0x00000000u)) {
      float a1 = lambdap[o];
      float b1 = lambdam[n];
      float lim1 = a1;
      if (lim1 > b1) {
        lim1 = b1;
      }
      lim = lim1;
    }
    float cur = lambda[((int)((int)(get_global_id(0))))];
    float _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[((int)((int)(get_global_id(0))))] = _out;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_mules_fill_lambda = R"kernel(
__kernel void navatala_cfd_vof_mules_fill_lambda(__global const int* counts, __global const float* paramsF, __global float* lambda) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    lambda[((int)((int)(get_global_id(0))))] = as_float(0x3f800000u);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_mules_init = R"kernel(
__kernel void navatala_cfd_vof_mules_init(__global const float* alpha, __global const float* alphaF, __global const float* phiBD, __global const float* phiCorr, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const int* owner, __global const int* nei, __global const float* vol, __global const int* counts, __global const float* paramsF, __global const float* rSubDeltaT, __global float* psiMaxCap, __global float* psiMinCap, __global float* sumPhip, __global float* mSumPhim) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)((int)(get_global_id(0))))];
    if (psi < as_float(0x00000000u)) {
      psi = as_float(0x00000000u);
    }
    if (psi > as_float(0x3f800000u)) {
      psi = as_float(0x3f800000u);
    }
    float maxN = as_float(0x00000000u);
    float minN = as_float(0x3f800000u);
    float sumPhiBD = as_float(0x00000000u);
    float sp = as_float(0x00000000u);
    float sm = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = as_float(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > as_float(0x00000000u)) {
          nbr = nei[f];
        }
        v = alpha[nbr];
      } else {
        v = alphaF[f];
      }
      if (v < as_float(0x00000000u)) {
        v = as_float(0x00000000u);
      }
      if (v > as_float(0x3f800000u)) {
        v = as_float(0x3f800000u);
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
        if (s > as_float(0x00000000u)) {
          if (pc > as_float(0x00000000u)) {
            sp = (sp + pc);
          } else {
            sm = (sm + (-pc));
          }
        } else {
          if (pc > as_float(0x00000000u)) {
            sm = (sm + pc);
          } else {
            sp = (sp + (-pc));
          }
        }
      } else {
        if (pc > as_float(0x00000000u)) {
          sp = (sp + pc);
        } else {
          sm = (sm + (-pc));
        }
      }
    }
    maxN = (maxN + paramsF[2]);
    if (maxN > as_float(0x3f800000u)) {
      maxN = as_float(0x3f800000u);
    }
    minN = (minN - paramsF[2]);
    if (minN < as_float(0x00000000u)) {
      minN = as_float(0x00000000u);
    }
    if (paramsF[3] > as_float(0x00000000u)) {
      float omSmooth = (as_float(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > as_float(0x3f800000u)) {
        maxN = as_float(0x3f800000u);
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < as_float(0x00000000u)) {
        minN = as_float(0x00000000u);
      }
    }
    sumPhip[((int)((int)(get_global_id(0))))] = sp;
    mSumPhim[((int)((int)(get_global_id(0))))] = sm;
    float V = vol[((int)((int)(get_global_id(0))))];
    psiMaxCap[((int)((int)(get_global_id(0))))] = (((V * rSubDeltaT[((int)((int)(get_global_id(0))))]) * (maxN - psi)) + sumPhiBD);
    psiMinCap[((int)((int)(get_global_id(0))))] = (((V * rSubDeltaT[((int)((int)(get_global_id(0))))]) * (psi - minN)) - sumPhiBD);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_phi_b_d_corr = R"kernel(
__kernel void navatala_cfd_vof_phi_b_d_corr(__global const float* alpha, __global const float* phiAll, __global const int* owner, __global const int* nei, __global const float* alphaF, __global const float* alphaPhiAll, __global const int* params, __global float* phiBD, __global float* phiCorr) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    if (((int)((int)(get_global_id(0)))) < params[1]) {
      float ph = phiAll[((int)((int)(get_global_id(0))))];
      int o = owner[((int)((int)(get_global_id(0))))];
      int n = nei[((int)((int)(get_global_id(0))))];
      float a = as_float(0x00000000u);
      if (ph >= as_float(0x00000000u)) {
        a = alpha[o];
      } else {
        a = alpha[n];
      }
      if (a < as_float(0x00000000u)) {
        a = as_float(0x00000000u);
      }
      if (a > as_float(0x3f800000u)) {
        a = as_float(0x3f800000u);
      }
      float pbd = (ph * a);
      phiBD[((int)((int)(get_global_id(0))))] = pbd;
      phiCorr[((int)((int)(get_global_id(0))))] = (alphaPhiAll[((int)((int)(get_global_id(0))))] - pbd);
    } else {
      phiBD[((int)((int)(get_global_id(0))))] = alphaPhiAll[((int)((int)(get_global_id(0))))];
      phiCorr[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_phir = R"kernel(
__kernel void navatala_cfd_vof_phir(__global const float* gx, __global const float* gy, __global const float* gz, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* phi, __global const float* magSf, __global const int* owner, __global const int* nei, __global const int* faceParams, __global const float* paramsF, __global float* outPhir) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= faceParams[0]) {
    return;
  } else {
    if (paramsF[0] == as_float(0x00000000u)) {
      outPhir[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      return;
    }
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float gox = gx[o];
    float goy = gy[o];
    float goz = gz[o];
    float gnx = gx[n];
    float gny = gy[n];
    float gnz = gz[n];
    float gfx = (as_float(0x3f000000u) * (gox + gnx));
    float gfy = (as_float(0x3f000000u) * (goy + gny));
    float gfz = (as_float(0x3f000000u) * (goz + gnz));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float dotSf = (((gfx * sfX[((int)((int)(get_global_id(0))))]) + (gfy * sfY[((int)((int)(get_global_id(0))))])) + (gfz * sfZ[((int)((int)(get_global_id(0))))]));
    float nHatf = (dotSf / (magg + paramsF[1]));
    float phic = ((paramsF[0] * fabs(phi[((int)((int)(get_global_id(0))))])) / magSf[((int)((int)(get_global_id(0))))]);
    outPhir[((int)((int)(get_global_id(0))))] = (phic * nHatf);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_vof_rho_phi_accumulate = R"kernel(
__kernel void navatala_cfd_vof_rho_phi_accumulate(__global const float* alphaPhi, __global const float* phi, __global float* rhoPhi, __global const int* counts, __global const float* paramsF) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    float acc = rhoPhi[((int)((int)(get_global_id(0))))];
    float ap = alphaPhi[((int)((int)(get_global_id(0))))];
    float ph = phi[((int)((int)(get_global_id(0))))];
    rhoPhi[((int)((int)(get_global_id(0))))] = (acc + (paramsF[2] * ((ap * paramsF[0]) + (ph * paramsF[1]))));
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_alpha_phi_all[] = {
  { "alphaPhiInt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiAll", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "alphaPhiAllOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_alpha_phi_all = {
  1,
  "navatala_cfd_vof_alpha_phi_all",
  "opencl",
  "navatala_cfd_vof_alpha_phi_all",
  "kernel:opencl:navatala_cfd_vof_alpha_phi_all",
  "abi-r1:opencl:navatala_cfd_vof_alpha_phi_all",
  "abi-r1:opencl:navatala_cfd_vof_alpha_phi_all",
  5,
  kAbiArgs_opencl_navatala_cfd_vof_alpha_phi_all
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_alpha_phi_int[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phir", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceParams", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outAlphaPhi", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_alpha_phi_int = {
  1,
  "navatala_cfd_vof_alpha_phi_int",
  "opencl",
  "navatala_cfd_vof_alpha_phi_int",
  "kernel:opencl:navatala_cfd_vof_alpha_phi_int",
  "abi-r1:opencl:navatala_cfd_vof_alpha_phi_int",
  "abi-r1:opencl:navatala_cfd_vof_alpha_phi_int",
  8,
  kAbiArgs_opencl_navatala_cfd_vof_alpha_phi_int
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_average_face_scalar_to_cell[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumMag", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outAlpha", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_average_face_scalar_to_cell = {
  1,
  "navatala_cfd_vof_average_face_scalar_to_cell",
  "opencl",
  "navatala_cfd_vof_average_face_scalar_to_cell",
  "kernel:opencl:navatala_cfd_vof_average_face_scalar_to_cell",
  "abi-r1:opencl:navatala_cfd_vof_average_face_scalar_to_cell",
  "abi-r1:opencl:navatala_cfd_vof_average_face_scalar_to_cell",
  7,
  kAbiArgs_opencl_navatala_cfd_vof_average_face_scalar_to_cell
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_interp_scalar_face_all[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcAlphaF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alphaF", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_interp_scalar_face_all = {
  1,
  "navatala_cfd_vof_interp_scalar_face_all",
  "opencl",
  "navatala_cfd_vof_interp_scalar_face_all",
  "kernel:opencl:navatala_cfd_vof_interp_scalar_face_all",
  "abi-r1:opencl:navatala_cfd_vof_interp_scalar_face_all",
  "abi-r1:opencl:navatala_cfd_vof_interp_scalar_face_all",
  7,
  kAbiArgs_opencl_navatala_cfd_vof_interp_scalar_face_all
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_alpha_update[] = {
  { "alpha", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divA", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rSubDeltaT", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaNew", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_alpha_update = {
  1,
  "navatala_cfd_vof_alpha_update",
  "opencl",
  "navatala_cfd_vof_alpha_update",
  "kernel:opencl:navatala_cfd_vof_alpha_update",
  "abi-r1:opencl:navatala_cfd_vof_alpha_update",
  "abi-r1:opencl:navatala_cfd_vof_alpha_update",
  6,
  kAbiArgs_opencl_navatala_cfd_vof_alpha_update
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_mules_apply[] = {
  { "phiBD", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCorr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "alphaPhi", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_mules_apply = {
  1,
  "navatala_cfd_vof_mules_apply",
  "opencl",
  "navatala_cfd_vof_mules_apply",
  "kernel:opencl:navatala_cfd_vof_mules_apply",
  "abi-r1:opencl:navatala_cfd_vof_mules_apply",
  "abi-r1:opencl:navatala_cfd_vof_mules_apply",
  6,
  kAbiArgs_opencl_navatala_cfd_vof_mules_apply
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_mules_cell_lambda[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_mules_cell_lambda = {
  1,
  "navatala_cfd_vof_mules_cell_lambda",
  "opencl",
  "navatala_cfd_vof_mules_cell_lambda",
  "kernel:opencl:navatala_cfd_vof_mules_cell_lambda",
  "abi-r1:opencl:navatala_cfd_vof_mules_cell_lambda",
  "abi-r1:opencl:navatala_cfd_vof_mules_cell_lambda",
  10,
  kAbiArgs_opencl_navatala_cfd_vof_mules_cell_lambda
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_mules_cell_sums[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_mules_cell_sums = {
  1,
  "navatala_cfd_vof_mules_cell_sums",
  "opencl",
  "navatala_cfd_vof_mules_cell_sums",
  "kernel:opencl:navatala_cfd_vof_mules_cell_sums",
  "abi-r1:opencl:navatala_cfd_vof_mules_cell_sums",
  "abi-r1:opencl:navatala_cfd_vof_mules_cell_sums",
  9,
  kAbiArgs_opencl_navatala_cfd_vof_mules_cell_sums
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_mules_face_update[] = {
  { "phiCorr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdam", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdap", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_mules_face_update = {
  1,
  "navatala_cfd_vof_mules_face_update",
  "opencl",
  "navatala_cfd_vof_mules_face_update",
  "kernel:opencl:navatala_cfd_vof_mules_face_update",
  "abi-r1:opencl:navatala_cfd_vof_mules_face_update",
  "abi-r1:opencl:navatala_cfd_vof_mules_face_update",
  8,
  kAbiArgs_opencl_navatala_cfd_vof_mules_face_update
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_mules_fill_lambda[] = {
  { "counts", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 16, 16, 256, nullptr, 0, 0 },
  { "lambda", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_mules_fill_lambda = {
  1,
  "navatala_cfd_vof_mules_fill_lambda",
  "opencl",
  "navatala_cfd_vof_mules_fill_lambda",
  "kernel:opencl:navatala_cfd_vof_mules_fill_lambda",
  "abi-r1:opencl:navatala_cfd_vof_mules_fill_lambda",
  "abi-r1:opencl:navatala_cfd_vof_mules_fill_lambda",
  3,
  kAbiArgs_opencl_navatala_cfd_vof_mules_fill_lambda
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_mules_init[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_mules_init = {
  1,
  "navatala_cfd_vof_mules_init",
  "opencl",
  "navatala_cfd_vof_mules_init",
  "kernel:opencl:navatala_cfd_vof_mules_init",
  "abi-r1:opencl:navatala_cfd_vof_mules_init",
  "abi-r1:opencl:navatala_cfd_vof_mules_init",
  17,
  kAbiArgs_opencl_navatala_cfd_vof_mules_init
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_phi_b_d_corr[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_phi_b_d_corr = {
  1,
  "navatala_cfd_vof_phi_b_d_corr",
  "opencl",
  "navatala_cfd_vof_phi_b_d_corr",
  "kernel:opencl:navatala_cfd_vof_phi_b_d_corr",
  "abi-r1:opencl:navatala_cfd_vof_phi_b_d_corr",
  "abi-r1:opencl:navatala_cfd_vof_phi_b_d_corr",
  9,
  kAbiArgs_opencl_navatala_cfd_vof_phi_b_d_corr
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_phir[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_phir = {
  1,
  "navatala_cfd_vof_phir",
  "opencl",
  "navatala_cfd_vof_phir",
  "kernel:opencl:navatala_cfd_vof_phir",
  "abi-r1:opencl:navatala_cfd_vof_phir",
  "abi-r1:opencl:navatala_cfd_vof_phir",
  13,
  kAbiArgs_opencl_navatala_cfd_vof_phir
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_vof_rho_phi_accumulate[] = {
  { "alphaPhi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoPhi", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_vof_rho_phi_accumulate = {
  1,
  "navatala_cfd_vof_rho_phi_accumulate",
  "opencl",
  "navatala_cfd_vof_rho_phi_accumulate",
  "kernel:opencl:navatala_cfd_vof_rho_phi_accumulate",
  "abi-r1:opencl:navatala_cfd_vof_rho_phi_accumulate",
  "abi-r1:opencl:navatala_cfd_vof_rho_phi_accumulate",
  5,
  kAbiArgs_opencl_navatala_cfd_vof_rho_phi_accumulate
};

bool tryGetKernelAbiManifest_opencl_cfd_vof(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_alpha_phi_all") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_alpha_phi_all;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_alpha_phi_int") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_alpha_phi_int;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_average_face_scalar_to_cell") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_average_face_scalar_to_cell;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_interp_scalar_face_all") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_interp_scalar_face_all;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_alpha_update") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_alpha_update;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_apply") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_mules_apply;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_cell_lambda") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_mules_cell_lambda;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_cell_sums") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_mules_cell_sums;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_face_update") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_mules_face_update;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_fill_lambda") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_mules_fill_lambda;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_init") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_mules_init;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_phi_b_d_corr") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_phi_b_d_corr;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_phir") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_phir;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_rho_phi_accumulate") {
    out = &kAbiManifest_opencl_navatala_cfd_vof_rho_phi_accumulate;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_cfd_vof(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_alpha_phi_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_alpha_phi_all";
    std::string_view sv(k_opencl_navatala_cfd_vof_alpha_phi_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_alpha_phi_int") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_alpha_phi_int";
    std::string_view sv(k_opencl_navatala_cfd_vof_alpha_phi_int);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_average_face_scalar_to_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_average_face_scalar_to_cell";
    std::string_view sv(k_opencl_navatala_cfd_vof_average_face_scalar_to_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_interp_scalar_face_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_interp_scalar_face_all";
    std::string_view sv(k_opencl_navatala_cfd_vof_interp_scalar_face_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_alpha_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_alpha_update";
    std::string_view sv(k_opencl_navatala_cfd_vof_alpha_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_apply") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_mules_apply";
    std::string_view sv(k_opencl_navatala_cfd_vof_mules_apply);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_cell_lambda") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_mules_cell_lambda";
    std::string_view sv(k_opencl_navatala_cfd_vof_mules_cell_lambda);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_cell_sums") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_mules_cell_sums";
    std::string_view sv(k_opencl_navatala_cfd_vof_mules_cell_sums);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_face_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_mules_face_update";
    std::string_view sv(k_opencl_navatala_cfd_vof_mules_face_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_fill_lambda") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_mules_fill_lambda";
    std::string_view sv(k_opencl_navatala_cfd_vof_mules_fill_lambda);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_mules_init") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_mules_init";
    std::string_view sv(k_opencl_navatala_cfd_vof_mules_init);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_phi_b_d_corr") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_phi_b_d_corr";
    std::string_view sv(k_opencl_navatala_cfd_vof_phi_b_d_corr);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_phir") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_phir";
    std::string_view sv(k_opencl_navatala_cfd_vof_phir);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_vof_rho_phi_accumulate") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_vof_rho_phi_accumulate";
    std::string_view sv(k_opencl_navatala_cfd_vof_rho_phi_accumulate);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

