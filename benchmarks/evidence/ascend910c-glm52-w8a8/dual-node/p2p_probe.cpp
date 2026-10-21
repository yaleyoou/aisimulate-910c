#include <acl/acl.h>
#include <hccl/hccl.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct LaunchGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool open = false;
};

void await_launch(void* argument) {
    auto* gate = static_cast<LaunchGate*>(argument);
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->changed.wait(lock, [gate]() { return gate->open; });
}

void require_success(int status, const std::string& operation) {
    if (status != 0) {
        throw std::runtime_error(operation + " failed: " + std::to_string(status));
    }
}

std::vector<int> parse_devices(const std::string& value) {
    std::stringstream stream(value);
    std::string token;
    std::vector<int> devices;
    while (std::getline(stream, token, ',')) {
        devices.push_back(std::stoi(token));
    }
    if (devices.empty()) {
        throw std::runtime_error("Empty device list");
    }
    return devices;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    try {
        if (argc != 6 || world < 2 || world % 2 != 0) {
            throw std::runtime_error("Usage: p2p_probe devices output_prefix iterations repeats max_bytes; even ranks >=2");
        }
        const auto devices = parse_devices(argv[1]);
        const std::string output_prefix = argv[2];
        const int iterations = std::stoi(argv[3]);
        const int repeats = std::stoi(argv[4]);
        const bool capture = std::getenv("AISIM_CAPTURE") != nullptr;
        const bool queued = std::getenv("AISIM_DEVICE_QUEUE") != nullptr;
        const size_t maximum_bytes = std::stoull(argv[5]);
        const char* local_rank_text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!local_rank_text || iterations <= 0 || repeats < 2) {
            throw std::runtime_error("OpenMPI local rank and valid sampling parameters are required");
        }
        const int local_rank = std::stoi(local_rank_text);
        if (local_rank >= static_cast<int>(devices.size())) {
            throw std::runtime_error("Local rank exceeds the explicit device map");
        }
        const int device = devices[local_rank];
        require_success(aclInit(nullptr), "aclInit");
        require_success(aclrtSetDevice(device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        require_success(aclrtCreateStream(&stream), "aclrtCreateStream");
        size_t free_before = 0;
        size_t total_bytes = 0;
        require_success(aclrtGetMemInfo(ACL_HBM_MEM, &free_before, &total_bytes), "aclrtGetMemInfo before");
        HcclRootInfo root_info{};
        if (rank == 0) {
            require_success(HcclGetRootInfo(&root_info), "HcclGetRootInfo");
        }
        MPI_Bcast(&root_info, sizeof(root_info), MPI_BYTE, 0, MPI_COMM_WORLD);
        HcclComm communicator = nullptr;
        require_success(HcclCommInitRootInfo(world, &root_info, rank, &communicator), "HcclCommInitRootInfo");
        const int half = world / 2;
        const bool initiator = rank < half;
        const int peer = initiator ? rank + half : rank - half;
        std::ofstream output(output_prefix + "-rank" + std::to_string(rank) + ".jsonl");
        if (!output) {
            throw std::runtime_error("Cannot create rank output");
        }
        output << std::setprecision(12);
        output << "{\"type\":\"metadata\",\"rank\":" << rank << ",\"world\":" << world
               << ",\"device\":" << device << ",\"peer\":" << peer
               << ",\"runtime_total_memory_bytes\":" << total_bytes
               << ",\"free_before_comm_bytes\":" << free_before
               << ",\"iterations\":" << iterations << ",\"repeats\":" << repeats
               << ",\"capture\":" << (capture ? "true" : "false")
               << ",\"device_queue\":" << (queued ? "true" : "false")
               << ",\"method\":\"HCCL ping-pong; each iteration sends the full payload in both directions; reported time is half round-trip\"}\n";
        output.flush();
        const std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384, 65536, 262144,
                                         1048576, 4194304, 16777216, 67108864, 134217728, 268435456};
        for (const auto bytes : sizes) {
            if (bytes > maximum_bytes) {
                break;
            }
            void* send_buffer = nullptr;
            void* receive_buffer = nullptr;
            require_success(aclrtMalloc(&send_buffer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc send");
            require_success(aclrtMalloc(&receive_buffer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc receive");
            const uint8_t pattern = static_cast<uint8_t>((rank % 251) + 1);
            std::vector<uint8_t> host(bytes, pattern);
            require_success(aclrtMemcpy(send_buffer, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload pattern");
            auto exchange = [&]() {
                if (initiator) {
                    require_success(HcclSend(send_buffer, bytes, HCCL_DATA_TYPE_UINT8, peer, communicator, stream), "HcclSend");
                    require_success(HcclRecv(receive_buffer, bytes, HCCL_DATA_TYPE_UINT8, peer, communicator, stream), "HcclRecv");
                } else {
                    require_success(HcclRecv(receive_buffer, bytes, HCCL_DATA_TYPE_UINT8, peer, communicator, stream), "HcclRecv");
                    require_success(HcclSend(receive_buffer, bytes, HCCL_DATA_TYPE_UINT8, peer, communicator, stream), "HcclSend");
                }
            };
            MPI_Barrier(MPI_COMM_WORLD);
            for (int warmup = 0; warmup < 50; ++warmup) {
                exchange();
            }
            require_success(aclrtSynchronizeStream(stream), "warmup synchronize");
            require_success(aclrtMemcpy(host.data(), bytes, receive_buffer, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "verify download");
            const uint8_t expected = initiator ? pattern : static_cast<uint8_t>((peer % 251) + 1);
            if (!std::all_of(host.begin(), host.end(), [expected](uint8_t value) { return value == expected; })) {
                throw std::runtime_error("Full payload verification failed");
            }
            aclmdlRI graph = nullptr;
            if (capture) {
                require_success(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture begin");
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    exchange();
                }
                require_success(aclmdlRICaptureEnd(stream, &graph), "capture end");
                for (int warmup = 0; warmup < 10; ++warmup) {
                    require_success(aclmdlRIExecuteAsync(graph, stream), "graph warmup");
                }
                require_success(aclrtSynchronizeStream(stream), "graph warmup synchronize");
            }
            for (int repeat = 0; repeat < repeats; ++repeat) {
                aclrtEvent begin = nullptr;
                aclrtEvent end = nullptr;
                require_success(aclrtCreateEventWithFlag(&begin, ACL_EVENT_TIME_LINE), "create begin event");
                require_success(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE), "create end event");
                MPI_Barrier(MPI_COMM_WORLD);
                const auto wall_start = std::chrono::steady_clock::now();
                LaunchGate gate;
                if (queued) {
                    require_success(aclrtLaunchHostFunc(stream, await_launch, &gate), "queue launch gate");
                }
                require_success(aclrtRecordEvent(begin, stream), "record begin");
                if (capture) {
                    require_success(aclmdlRIExecuteAsync(graph, stream), "graph execute");
                } else {
                    for (int iteration = 0; iteration < iterations; ++iteration) {
                        exchange();
                    }
                }
                require_success(aclrtRecordEvent(end, stream), "record end");
                if (queued) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    {
                        std::lock_guard<std::mutex> lock(gate.mutex);
                        gate.open = true;
                    }
                    gate.changed.notify_one();
                }
                require_success(aclrtSynchronizeStream(stream), "sample synchronize");
                const auto wall_end = std::chrono::steady_clock::now();
                float device_ms = 0;
                require_success(aclrtEventElapsedTime(&device_ms, begin, end), "elapsed event time");
                const double divisor = 2.0 * iterations;
                const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count() / divisor;
                output << "{\"type\":\"sample\",\"rank\":" << rank << ",\"bytes\":" << bytes
                       << ",\"repeat\":" << repeat << ",\"device_ms_per_direction\":" << device_ms / divisor
                       << ",\"wall_ms_per_direction\":" << wall_ms << ",\"correctness\":true}\n";
                output.flush();
                require_success(aclrtDestroyEvent(begin), "destroy begin event");
                require_success(aclrtDestroyEvent(end), "destroy end event");
            }
            require_success(aclrtMemcpy(host.data(), bytes, receive_buffer, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "post-timing verify download");
            if (!std::all_of(host.begin(), host.end(), [expected](uint8_t value) { return value == expected; })) {
                throw std::runtime_error("Post-timing full payload verification failed");
            }
            output << "{\"type\":\"post_timing_verification\",\"rank\":" << rank
                   << ",\"bytes\":" << bytes << ",\"correctness\":true}\n";
            output.flush();
            if (capture) {
                require_success(aclmdlRIDestroy(graph), "destroy graph");
            }
            require_success(aclrtFree(send_buffer), "free send buffer");
            require_success(aclrtFree(receive_buffer), "free receive buffer");
            if (rank == 0) {
                std::cout << "completed bytes=" << bytes << " ranks=" << world << std::endl;
            }
        }
        size_t free_after = 0;
        require_success(aclrtGetMemInfo(ACL_HBM_MEM, &free_after, &total_bytes), "aclrtGetMemInfo after");
        output << "{\"type\":\"complete\",\"free_after_comm_bytes\":" << free_after << "}\n";
        output.close();
        MPI_Barrier(MPI_COMM_WORLD);
        require_success(HcclCommDestroy(communicator), "HcclCommDestroy");
        require_success(aclrtDestroyStream(stream), "aclrtDestroyStream");
        require_success(aclrtResetDevice(device), "aclrtResetDevice");
        require_success(aclFinalize(), "aclFinalize");
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank=" << rank << " " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
