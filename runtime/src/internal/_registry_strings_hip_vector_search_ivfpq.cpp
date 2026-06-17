// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_vector_search_ivfpq` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_vector_search_ivfpq_compute_centroid_distances_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_compute_centroid_distances_f32(const float* queries, const float* centroids, const unsigned int* n_queries, const unsigned int* n_lists, const unsigned int* dim, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_q = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_c = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nq = n_queries[0];
  unsigned int nl = n_lists[0];
  unsigned int d = dim[0];
  bool in_q = (gid_q < nq);
  bool in_c = (gid_c < nl);
  bool inBounds = (in_q && in_c);
  if (inBounds) {
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((gid_q * d) + k);
      unsigned int c_idx = ((gid_c * d) + k);
      float q_val = queries[q_idx];
      float c_val = centroids[c_idx];
      float diff = (q_val - c_val);
      float diff_sq = (diff * diff);
      float old = acc;
      acc = (old + diff_sq);
    }
    unsigned int out_idx = ((gid_q * nl) + gid_c);
    float result = acc;
    distances[out_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32(const float* query, const float* codebook, const unsigned int* n_subspaces, const unsigned int* n_codes, const unsigned int* sub_dim, float* lut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int ns = n_subspaces[0];
  unsigned int nc = n_codes[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (ns * nc);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int subspace = (gid / nc);
    unsigned int code = (gid % nc);
    unsigned int q_start = (subspace * sd);
    unsigned int cb_base = ((subspace * (nc * sd)) + (code * sd));
    float dist = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(sd); ++k) {
      unsigned int q_idx = (q_start + k);
      unsigned int cb_idx = (cb_base + k);
      float q_val = query[q_idx];
      float cb_val = codebook[cb_idx];
      float diff = (q_val - cb_val);
      float diff_sq = (diff * diff);
      float old = dist;
      dist = (old + diff_sq);
    }
    float result = dist;
    lut[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_ivfpq_compute_p_q_distances_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_compute_p_q_distances_f32(const unsigned char* pq_codes, const float* lut, const unsigned int* vector_indices, const unsigned int* list_start, const unsigned int* list_size, const unsigned int* n_subspaces, const unsigned int* n_codes, float* distances, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int start = list_start[0];
  unsigned int size = list_size[0];
  unsigned int ns = n_subspaces[0];
  unsigned int nc = n_codes[0];
  bool inBounds = (gid < size);
  if (inBounds) {
    unsigned int vec_idx = (start + gid);
    float dist = __uint_as_float(0x00000000u);
    for (int s = 0; s < (int)(ns); ++s) {
      unsigned int code_idx = ((vec_idx * ns) + s);
      unsigned char code_u8 = pq_codes[code_idx];
      unsigned int code = ((unsigned int)(code_u8));
      unsigned int lut_idx = ((s * nc) + code);
      float lut_val = lut[lut_idx];
      float old = dist;
      dist = (old + lut_val);
    }
    unsigned int orig_idx = vector_indices[vec_idx];
    float result = dist;
    distances[gid] = result;
    indices[gid] = orig_idx;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_ivfpq_encode_p_q_codes = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_encode_p_q_codes(const float* vectors, const float* codebook, const unsigned int* n_vectors, const unsigned int* n_subspaces, const unsigned int* n_codes, const unsigned int* sub_dim, unsigned char* codes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int ns = n_subspaces[0];
  unsigned int nc = n_codes[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (nv * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int vec_idx = (gid / ns);
    unsigned int subspace = (gid % ns);
    unsigned int dim = (ns * sd);
    unsigned int v_start = ((vec_idx * dim) + (subspace * sd));
    unsigned int cb_subspace_base = (subspace * (nc * sd));
    unsigned int best_code = 0u;
    float best_dist = __uint_as_float(0x7e967699u);
    for (int c = 0; c < (int)(nc); ++c) {
      unsigned int cb_base = (cb_subspace_base + (c * sd));
      float dist = __uint_as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        unsigned int v_idx = (v_start + k);
        unsigned int cb_idx = (cb_base + k);
        float v_val = vectors[v_idx];
        float cb_val = codebook[cb_idx];
        float diff = (v_val - cb_val);
        float diff_sq = (diff * diff);
        float old_d = dist;
        dist = (old_d + diff_sq);
      }
      float curr_dist = dist;
      float curr_best = best_dist;
      bool is_better = (curr_dist < curr_best);
      if (is_better) {
        best_dist = curr_dist;
        best_code = c;
      }
    }
    unsigned int final_code = best_code;
    unsigned char code_u8 = ((unsigned char)(final_code));
    codes[gid] = code_u8;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_ivfpq_compute_residuals_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_compute_residuals_f32(const float* vectors, const float* centroids, const unsigned int* assignments, const unsigned int* n_vectors, const unsigned int* dim, float* residuals) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  unsigned int total = (nv * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int vec_idx = (gid / d);
    unsigned int dim_idx = (gid % d);
    unsigned int cluster = assignments[vec_idx];
    float v_val = vectors[gid];
    unsigned int c_idx = ((cluster * d) + dim_idx);
    float c_val = centroids[c_idx];
    float residual = (v_val - c_val);
    residuals[gid] = residual;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_ivfpq_compute_centroid_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_ivfpq_compute_centroid_distances_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "hip",
  "navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "kernel:hip:navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_ivfpq_compute_centroid_distances_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_subspaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "lut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "hip",
  "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "kernel:hip:navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_ivfpq_compute_p_q_distances_f32[] = {
  { "pq_codes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lut", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vector_indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_start", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "list_size", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_subspaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codes", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_ivfpq_compute_p_q_distances_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "hip",
  "navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "kernel:hip:navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  9,
  kAbiArgs_hip_navatala_vector_search_ivfpq_compute_p_q_distances_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_ivfpq_encode_p_q_codes[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_subspaces", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codes", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codes", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_ivfpq_encode_p_q_codes = {
  1,
  "navatala_vector_search_ivfpq_encode_p_q_codes",
  "hip",
  "navatala_vector_search_ivfpq_encode_p_q_codes",
  "kernel:hip:navatala_vector_search_ivfpq_encode_p_q_codes",
  "abi-r1:hip:navatala_vector_search_ivfpq_encode_p_q_codes",
  "abi-r1:hip:navatala_vector_search_ivfpq_encode_p_q_codes",
  7,
  kAbiArgs_hip_navatala_vector_search_ivfpq_encode_p_q_codes
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_ivfpq_compute_residuals_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "residuals", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_ivfpq_compute_residuals_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_residuals_f32",
  "hip",
  "navatala_vector_search_ivfpq_compute_residuals_f32",
  "kernel:hip:navatala_vector_search_ivfpq_compute_residuals_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_residuals_f32",
  "abi-r1:hip:navatala_vector_search_ivfpq_compute_residuals_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_ivfpq_compute_residuals_f32
};

bool tryGetKernelAbiManifest_hip_vector_search_ivfpq(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_centroid_distances_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_ivfpq_compute_centroid_distances_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_distances_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_ivfpq_compute_p_q_distances_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_encode_p_q_codes") {
    out = &kAbiManifest_hip_navatala_vector_search_ivfpq_encode_p_q_codes;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_residuals_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_ivfpq_compute_residuals_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_vector_search_ivfpq(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_centroid_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_centroid_distances_f32";
    std::string_view sv(k_hip_navatala_vector_search_ivfpq_compute_centroid_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32";
    std::string_view sv(k_hip_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_p_q_distances_f32";
    std::string_view sv(k_hip_navatala_vector_search_ivfpq_compute_p_q_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_encode_p_q_codes") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_ivfpq_encode_p_q_codes";
    std::string_view sv(k_hip_navatala_vector_search_ivfpq_encode_p_q_codes);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_ivfpq_compute_residuals_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_residuals_f32";
    std::string_view sv(k_hip_navatala_vector_search_ivfpq_compute_residuals_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

