// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_vector_search_ivfpq` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_vector_search_ivfpq_compute_centroid_distances_f32 = R"kernel(
__kernel void navatala_vector_search_ivfpq_compute_centroid_distances_f32(__global const float* queries, __global const float* centroids, __global const uint* n_queries, __global const uint* n_lists, __global const uint* dim, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid_q = ((uint)((int)(get_global_id(0))));
  uint gid_c = ((uint)((int)(get_global_id(1))));
  uint nq = n_queries[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  bool in_q = (gid_q < nq);
  bool in_c = (gid_c < nl);
  bool inBounds = (in_q && in_c);
  if (inBounds) {
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((gid_q * d) + k);
      uint c_idx = ((gid_c * d) + k);
      float q_val = queries[q_idx];
      float c_val = centroids[c_idx];
      float diff = (q_val - c_val);
      float diff_sq = (diff * diff);
      float old = acc;
      acc = (old + diff_sq);
    }
    uint out_idx = ((gid_q * nl) + gid_c);
    float result = acc;
    distances[out_idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32 = R"kernel(
__kernel void navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32(__global const float* query, __global const float* codebook, __global const uint* n_subspaces, __global const uint* n_codes, __global const uint* sub_dim, __global float* lut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  uint sd = sub_dim[0];
  uint total = (ns * nc);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint subspace = (gid / nc);
    uint code = (gid % nc);
    uint q_start = (subspace * sd);
    uint cb_base = ((subspace * (nc * sd)) + (code * sd));
    float dist = as_float(0x00000000u);
    for (int k = 0; k < (int)(sd); ++k) {
      uint q_idx = (q_start + k);
      uint cb_idx = (cb_base + k);
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
const char* k_opencl_navatala_vector_search_ivfpq_compute_p_q_distances_f32 = R"kernel(
__kernel void navatala_vector_search_ivfpq_compute_p_q_distances_f32(__global const uchar* pq_codes, __global const float* lut, __global const uint* vector_indices, __global const uint* list_start, __global const uint* list_size, __global const uint* n_subspaces, __global const uint* n_codes, __global float* distances, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint start = list_start[0];
  uint size = list_size[0];
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  bool inBounds = (gid < size);
  if (inBounds) {
    uint vec_idx = (start + gid);
    float dist = as_float(0x00000000u);
    for (int s = 0; s < (int)(ns); ++s) {
      uint code_idx = ((vec_idx * ns) + s);
      uchar code_u8 = pq_codes[code_idx];
      uint code = ((uint)(code_u8));
      uint lut_idx = ((s * nc) + code);
      float lut_val = lut[lut_idx];
      float old = dist;
      dist = (old + lut_val);
    }
    uint orig_idx = vector_indices[vec_idx];
    float result = dist;
    distances[gid] = result;
    indices[gid] = orig_idx;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_ivfpq_encode_p_q_codes = R"kernel(
__kernel void navatala_vector_search_ivfpq_encode_p_q_codes(__global const float* vectors, __global const float* codebook, __global const uint* n_vectors, __global const uint* n_subspaces, __global const uint* n_codes, __global const uint* sub_dim, __global uchar* codes) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  uint sd = sub_dim[0];
  uint total = (nv * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint vec_idx = (gid / ns);
    uint subspace = (gid % ns);
    uint dim = (ns * sd);
    uint v_start = ((vec_idx * dim) + (subspace * sd));
    uint cb_subspace_base = (subspace * (nc * sd));
    uint best_code = (uint)(0u);
    float best_dist = as_float(0x7e967699u);
    for (int c = 0; c < (int)(nc); ++c) {
      uint cb_base = (cb_subspace_base + (c * sd));
      float dist = as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        uint v_idx = (v_start + k);
        uint cb_idx = (cb_base + k);
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
    uint final_code = best_code;
    uchar code_u8 = ((uchar)(final_code));
    codes[gid] = code_u8;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_ivfpq_compute_residuals_f32 = R"kernel(
__kernel void navatala_vector_search_ivfpq_compute_residuals_f32(__global const float* vectors, __global const float* centroids, __global const uint* assignments, __global const uint* n_vectors, __global const uint* dim, __global float* residuals) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint vec_idx = (gid / d);
    uint dim_idx = (gid % d);
    uint cluster = assignments[vec_idx];
    float v_val = vectors[gid];
    uint c_idx = ((cluster * d) + dim_idx);
    float c_val = centroids[c_idx];
    float residual = (v_val - c_val);
    residuals[gid] = residual;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_centroid_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_centroid_distances_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "opencl",
  "navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "kernel:opencl:navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_centroid_distances_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_centroid_distances_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_subspaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "lut", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "opencl",
  "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "kernel:opencl:navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_p_q_distances_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_p_q_distances_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "opencl",
  "navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "kernel:opencl:navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_p_q_distances_f32",
  9,
  kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_p_q_distances_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_ivfpq_encode_p_q_codes[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_subspaces", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codes", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codes", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_ivfpq_encode_p_q_codes = {
  1,
  "navatala_vector_search_ivfpq_encode_p_q_codes",
  "opencl",
  "navatala_vector_search_ivfpq_encode_p_q_codes",
  "kernel:opencl:navatala_vector_search_ivfpq_encode_p_q_codes",
  "abi-r1:opencl:navatala_vector_search_ivfpq_encode_p_q_codes",
  "abi-r1:opencl:navatala_vector_search_ivfpq_encode_p_q_codes",
  7,
  kAbiArgs_opencl_navatala_vector_search_ivfpq_encode_p_q_codes
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_residuals_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "residuals", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_residuals_f32 = {
  1,
  "navatala_vector_search_ivfpq_compute_residuals_f32",
  "opencl",
  "navatala_vector_search_ivfpq_compute_residuals_f32",
  "kernel:opencl:navatala_vector_search_ivfpq_compute_residuals_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_residuals_f32",
  "abi-r1:opencl:navatala_vector_search_ivfpq_compute_residuals_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_ivfpq_compute_residuals_f32
};

bool tryGetKernelAbiManifest_opencl_vector_search_ivfpq(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_centroid_distances_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_centroid_distances_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_distances_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_p_q_distances_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_encode_p_q_codes") {
    out = &kAbiManifest_opencl_navatala_vector_search_ivfpq_encode_p_q_codes;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_residuals_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_ivfpq_compute_residuals_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_vector_search_ivfpq(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_centroid_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_centroid_distances_f32";
    std::string_view sv(k_opencl_navatala_vector_search_ivfpq_compute_centroid_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32";
    std::string_view sv(k_opencl_navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_p_q_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_p_q_distances_f32";
    std::string_view sv(k_opencl_navatala_vector_search_ivfpq_compute_p_q_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_encode_p_q_codes") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_ivfpq_encode_p_q_codes";
    std::string_view sv(k_opencl_navatala_vector_search_ivfpq_encode_p_q_codes);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_ivfpq_compute_residuals_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_ivfpq_compute_residuals_f32";
    std::string_view sv(k_opencl_navatala_vector_search_ivfpq_compute_residuals_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

