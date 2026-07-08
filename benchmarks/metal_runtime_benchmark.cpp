// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Metal runtime benchmark for public navatala_gpu releases.
//
// This intentionally uses only the stable gpu_runtime API plus inline MSL so
// it can run on a Metal-only generated corpus. It measures the runtime paths
// optimized by the Metal backend: repeated kernel submit/synchronize, batched
// kernel submits, and private-buffer blit copies.

#include "gpu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace GpuRuntime;

namespace {

struct Config {
    std::string jsonPath;
    std::size_t elements = 1u << 20;
    int iterations = 100;
    int warmup = 10;
    int burstSubmits = 32;
    int copyRepeats = 16;
};

struct Row {
    std::string operation;
    int iterations = 0;
    int innerRepeats = 0;
    double totalMs = 0.0;
    double meanMs = 0.0;
    double throughputGBps = 0.0;
    bool passed = false;
    double checksum = 0.0;
    std::string notes;
};

[[noreturn]] void usage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --json PATH          Write benchmark JSON.\n"
        << "  --elements N         Float element count (default 1048576).\n"
        << "  --iterations N       Timed outer iterations (default 100).\n"
        << "  --warmup N           Warmup iterations (default 10).\n"
        << "  --burst N            Kernel submits per burst iteration (default 32).\n"
        << "  --copy-repeats N     Copy roundtrips per copy iteration (default 16).\n"
        << "  --quick              Smaller smoke-sized run.\n"
        << "  -h, --help           Show this help.\n";
    std::exit(2);
}

std::size_t parseSize(const char* value, const char* flag)
{
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0' || parsed == 0) {
        throw std::runtime_error(std::string("invalid value for ") + flag + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

int parseInt(const char* value, const char* flag)
{
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid value for ") + flag + ": " + value);
    }
    return static_cast<int>(parsed);
}

Config parseArgs(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto requireValue = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };
        if (arg == "--json") {
            cfg.jsonPath = requireValue("--json");
        } else if (arg.rfind("--json=", 0) == 0) {
            cfg.jsonPath = arg.substr(std::strlen("--json="));
        } else if (arg == "--elements") {
            cfg.elements = parseSize(requireValue("--elements"), "--elements");
        } else if (arg.rfind("--elements=", 0) == 0) {
            cfg.elements = parseSize(arg.c_str() + std::strlen("--elements="), "--elements");
        } else if (arg == "--iterations") {
            cfg.iterations = parseInt(requireValue("--iterations"), "--iterations");
        } else if (arg.rfind("--iterations=", 0) == 0) {
            cfg.iterations = parseInt(arg.c_str() + std::strlen("--iterations="), "--iterations");
        } else if (arg == "--warmup") {
            cfg.warmup = parseInt(requireValue("--warmup"), "--warmup");
        } else if (arg.rfind("--warmup=", 0) == 0) {
            cfg.warmup = parseInt(arg.c_str() + std::strlen("--warmup="), "--warmup");
        } else if (arg == "--burst") {
            cfg.burstSubmits = parseInt(requireValue("--burst"), "--burst");
        } else if (arg.rfind("--burst=", 0) == 0) {
            cfg.burstSubmits = parseInt(arg.c_str() + std::strlen("--burst="), "--burst");
        } else if (arg == "--copy-repeats") {
            cfg.copyRepeats = parseInt(requireValue("--copy-repeats"), "--copy-repeats");
        } else if (arg.rfind("--copy-repeats=", 0) == 0) {
            cfg.copyRepeats = parseInt(arg.c_str() + std::strlen("--copy-repeats="), "--copy-repeats");
        } else if (arg == "--quick") {
            cfg.elements = 1u << 16;
            cfg.iterations = 10;
            cfg.warmup = 2;
            cfg.burstSubmits = 8;
            cfg.copyRepeats = 8;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    return cfg;
}

const char* envOrEmpty(const char* name)
{
    const char* value = std::getenv(name);
    return value ? value : "";
}

bool envEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && std::string(value) != "0";
}

std::string jsonEscape(const std::string& value)
{
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    return out.str();
}

double nowMs()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

void fillFloat(Buffer& buffer, std::size_t count, float scale, float bias)
{
    float* ptr = static_cast<float*>(buffer.mapPtr(MapMode::Write));
    if (!ptr) {
        throw std::runtime_error("failed to map float buffer");
    }
    for (std::size_t i = 0; i < count; ++i) {
        ptr[i] = bias + scale * static_cast<float>((i % 4096) + 1);
    }
    buffer.unmap();
}

void fillScalar(Buffer& buffer, const void* value, std::size_t bytes)
{
    void* ptr = buffer.mapPtr(MapMode::Write);
    if (!ptr) {
        throw std::runtime_error("failed to map scalar buffer");
    }
    std::memcpy(ptr, value, bytes);
    buffer.unmap();
}

double checksumFloat(Buffer& buffer, std::size_t count, std::size_t samples = 4096)
{
    const float* ptr = static_cast<const float*>(buffer.mapPtr(MapMode::Read));
    if (!ptr) {
        throw std::runtime_error("failed to map output buffer");
    }
    const std::size_t stride = std::max<std::size_t>(1, count / samples);
    double sum = 0.0;
    std::size_t visited = 0;
    for (std::size_t i = 0; i < count; i += stride) {
        const float v = ptr[i];
        if (!std::isfinite(v)) {
            buffer.unmap();
            throw std::runtime_error("non-finite output detected");
        }
        sum += static_cast<double>(v);
        ++visited;
    }
    buffer.unmap();
    return visited ? sum / static_cast<double>(visited) : 0.0;
}

std::unique_ptr<Program> makeProgram(Device& device)
{
    static const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void saxpy_step(device float* y [[buffer(0)]],
                       const device float* x [[buffer(1)]],
                       const device float* coeff [[buffer(2)]],
                       const device uint* nbuf [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    const uint n = nbuf[0];
    if (gid < n) {
        y[gid] = y[gid] + coeff[0] * x[gid];
    }
}
)MSL";

    ProgramSource source;
    source.kind = ProgramSource::Kind::Msl;
    source.entryPoint = "saxpy_step";
    source.bytes.assign(kSource, kSource + std::strlen(kSource));
    auto program = device.createProgram(source);
    if (!program) {
        throw std::runtime_error("failed to create Metal benchmark program");
    }
    return program;
}

struct SaxpyBuffers {
    std::unique_ptr<Buffer> hX;
    std::unique_ptr<Buffer> hY;
    std::unique_ptr<Buffer> hCoeff;
    std::unique_ptr<Buffer> hN;
    std::unique_ptr<Buffer> dX;
    std::unique_ptr<Buffer> dY;
    std::unique_ptr<Buffer> dCoeff;
    std::unique_ptr<Buffer> dN;
    std::unique_ptr<Buffer> hOut;
};

SaxpyBuffers makeSaxpyBuffers(Device& device, Queue& queue, std::size_t elements)
{
    const std::size_t bytes = elements * sizeof(float);
    SaxpyBuffers b;
    b.hX = device.createBuffer(bytes, MemoryKind::HostPinned);
    b.hY = device.createBuffer(bytes, MemoryKind::HostPinned);
    b.hCoeff = device.createBuffer(sizeof(float), MemoryKind::HostPinned);
    b.hN = device.createBuffer(sizeof(std::uint32_t), MemoryKind::HostPinned);
    b.dX = device.createBuffer(bytes, MemoryKind::Device);
    b.dY = device.createBuffer(bytes, MemoryKind::Device);
    b.dCoeff = device.createBuffer(sizeof(float), MemoryKind::Device);
    b.dN = device.createBuffer(sizeof(std::uint32_t), MemoryKind::Device);
    b.hOut = device.createBuffer(bytes, MemoryKind::HostPinned);
    if (!b.hX || !b.hY || !b.hCoeff || !b.hN || !b.dX || !b.dY || !b.dCoeff || !b.dN || !b.hOut) {
        throw std::runtime_error("failed to allocate saxpy buffers");
    }

    fillFloat(*b.hX, elements, 1.0e-5f, 1.0f);
    fillFloat(*b.hY, elements, 2.0e-5f, 0.25f);
    const float coeff = 1.0e-6f;
    const std::uint32_t n = static_cast<std::uint32_t>(elements);
    fillScalar(*b.hCoeff, &coeff, sizeof(coeff));
    fillScalar(*b.hN, &n, sizeof(n));

    queue.memcpy(*b.dX, *b.hX, bytes);
    queue.memcpy(*b.dY, *b.hY, bytes);
    queue.memcpy(*b.dCoeff, *b.hCoeff, sizeof(float));
    queue.memcpy(*b.dN, *b.hN, sizeof(std::uint32_t));
    queue.synchronize();
    return b;
}

Row runSubmitCase(Device& device,
                  Queue& queue,
                  Program& program,
                  const Config& cfg,
                  const std::string& operation,
                  int submitsPerIteration)
{
    auto buffers = makeSaxpyBuffers(device, queue, cfg.elements);
    const std::uint32_t block = 256;
    const std::uint32_t grid = static_cast<std::uint32_t>((cfg.elements + block - 1) / block);
    std::vector<Buffer*> args = {
        buffers.dY.get(),
        buffers.dX.get(),
        buffers.dCoeff.get(),
        buffers.dN.get(),
    };

    auto runLoop = [&](int iterations) {
        for (int i = 0; i < iterations; ++i) {
            for (int j = 0; j < submitsPerIteration; ++j) {
                queue.submit(program, args, grid, 1, 1, block, 1, 1);
            }
            queue.synchronize();
        }
    };

    runLoop(cfg.warmup);
    const double start = nowMs();
    runLoop(cfg.iterations);
    const double elapsed = nowMs() - start;

    queue.memcpy(*buffers.hOut, *buffers.dY, cfg.elements * sizeof(float));
    queue.synchronize();
    const double checksum = checksumFloat(*buffers.hOut, cfg.elements);

    Row row;
    row.operation = operation;
    row.iterations = cfg.iterations;
    row.innerRepeats = submitsPerIteration;
    row.totalMs = elapsed;
    row.meanMs = elapsed / static_cast<double>(cfg.iterations);
    row.passed = std::isfinite(checksum) && checksum > 0.0;
    row.checksum = checksum;
    row.notes = "kernel submits per outer iteration";
    return row;
}

Row runCopyCase(Device& device, Queue& queue, const Config& cfg)
{
    const std::size_t bytes = cfg.elements * sizeof(float);
    auto hIn = device.createBuffer(bytes, MemoryKind::HostPinned);
    auto hOut = device.createBuffer(bytes, MemoryKind::HostPinned);
    auto dev = device.createBuffer(bytes, MemoryKind::Device);
    if (!hIn || !hOut || !dev) {
        throw std::runtime_error("failed to allocate copy buffers");
    }
    fillFloat(*hIn, cfg.elements, 0.25f, 3.0f);

    auto runLoop = [&](int iterations) {
        for (int i = 0; i < iterations; ++i) {
            for (int j = 0; j < cfg.copyRepeats; ++j) {
                queue.memcpy(*dev, *hIn, bytes);
                queue.memcpy(*hOut, *dev, bytes);
            }
            queue.synchronize();
        }
    };

    runLoop(cfg.warmup);
    const double start = nowMs();
    runLoop(cfg.iterations);
    const double elapsed = nowMs() - start;
    const double checksum = checksumFloat(*hOut, cfg.elements);

    const double movedBytes = static_cast<double>(bytes) * 2.0 *
                              static_cast<double>(cfg.copyRepeats) *
                              static_cast<double>(cfg.iterations);
    Row row;
    row.operation = "copy_roundtrip";
    row.iterations = cfg.iterations;
    row.innerRepeats = cfg.copyRepeats;
    row.totalMs = elapsed;
    row.meanMs = elapsed / static_cast<double>(cfg.iterations);
    row.throughputGBps = elapsed > 0.0 ? movedBytes / (elapsed / 1000.0) / 1.0e9 : 0.0;
    row.passed = std::isfinite(checksum) && checksum > 0.0;
    row.checksum = checksum;
    row.notes = "H2D+D2H roundtrips per outer iteration";
    return row;
}

void writeJson(const Config& cfg, const Device& device, const std::vector<Row>& rows)
{
    if (cfg.jsonPath.empty()) {
        return;
    }
    std::ofstream out(cfg.jsonPath);
    if (!out) {
        throw std::runtime_error("failed to open JSON output: " + cfg.jsonPath);
    }

    out << std::setprecision(10);
    out << "{\n";
    out << "  \"schema\": \"navatala_gpu.metal_runtime_benchmark.v1\",\n";
    out << "  \"timingMode\": \"host_wall_synchronized_runtime_api\",\n";
    out << "  \"backend\": \"metal\",\n";
    out << "  \"device\": {\n";
    out << "    \"name\": \"" << jsonEscape(device.getName()) << "\",\n";
    out << "    \"computeCapability\": \"" << jsonEscape(device.getComputeCapability()) << "\",\n";
    out << "    \"totalMemoryBytes\": " << device.getTotalMemory() << "\n";
    out << "  },\n";
    out << "  \"config\": {\n";
    out << "    \"elements\": " << cfg.elements << ",\n";
    out << "    \"iterations\": " << cfg.iterations << ",\n";
    out << "    \"warmup\": " << cfg.warmup << ",\n";
    out << "    \"burstSubmits\": " << cfg.burstSubmits << ",\n";
    out << "    \"copyRepeats\": " << cfg.copyRepeats << "\n";
    out << "  },\n";
    out << "  \"env\": {\n";
    out << "    \"batchSubmits\": " << (envEnabled("NAVATALA_GPU_METAL_BATCH_SUBMITS") ? "true" : "false") << ",\n";
    out << "    \"batchLimit\": \"" << jsonEscape(envOrEmpty("NAVATALA_GPU_METAL_BATCH_LIMIT")) << "\",\n";
    out << "    \"batchBlits\": " << (envEnabled("NAVATALA_GPU_METAL_BATCH_BLITS") ? "true" : "false") << ",\n";
    out << "    \"batchBlitLimit\": \"" << jsonEscape(envOrEmpty("NAVATALA_GPU_METAL_BATCH_BLIT_LIMIT")) << "\",\n";
    out << "    \"privateDeviceBuffers\": " << (envEnabled("NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS") ? "true" : "false") << ",\n";
    out << "    \"privateMinBytes\": \"" << jsonEscape(envOrEmpty("NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES")) << "\"\n";
    out << "  },\n";
    out << "  \"rows\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Row& row = rows[i];
        out << "    {\n";
        out << "      \"operation\": \"" << jsonEscape(row.operation) << "\",\n";
        out << "      \"iterations\": " << row.iterations << ",\n";
        out << "      \"innerRepeats\": " << row.innerRepeats << ",\n";
        out << "      \"totalMs\": " << row.totalMs << ",\n";
        out << "      \"meanMs\": " << row.meanMs << ",\n";
        out << "      \"throughputGBps\": " << row.throughputGBps << ",\n";
        out << "      \"passed\": " << (row.passed ? "true" : "false") << ",\n";
        out << "      \"checksum\": " << row.checksum << ",\n";
        out << "      \"notes\": \"" << jsonEscape(row.notes) << "\"\n";
        out << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        setenv("GPU_RUNTIME_BACKEND", "metal", 1);
        Config cfg = parseArgs(argc, argv);

        auto device = Device::create(0);
        if (!device) {
            throw std::runtime_error("no Metal device available");
        }
        auto queue = device->createQueue(StreamPriority::Normal);
        if (!queue) {
            throw std::runtime_error("failed to create queue");
        }
        auto program = makeProgram(*device);

        std::vector<Row> rows;
        rows.push_back(runSubmitCase(*device, *queue, *program, cfg, "submit_sync", 1));
        rows.push_back(runSubmitCase(*device, *queue, *program, cfg, "submit_burst", cfg.burstSubmits));
        rows.push_back(runCopyCase(*device, *queue, cfg));

        bool ok = true;
        for (const Row& row : rows) {
            ok = ok && row.passed;
            std::cout << row.operation
                      << " mean_ms=" << row.meanMs
                      << " repeats=" << row.innerRepeats
                      << " checksum=" << row.checksum;
            if (row.throughputGBps > 0.0) {
                std::cout << " throughput_GBps=" << row.throughputGBps;
            }
            std::cout << (row.passed ? " PASS" : " FAIL") << "\n";
        }

        writeJson(cfg, *device, rows);
        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
