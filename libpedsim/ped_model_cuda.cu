#include "ped_model.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Ped {

__global__ void updateAgentPos(int size, float* xs, float* ys, float* dest_xs,
    float* dest_ys, float* dest_rs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff_x = dest_xs[idx] - xs[idx];
        float diff_y = dest_ys[idx] - ys[idx];
        float len = sqrtf(diff_x * diff_x + diff_y * diff_y);
        xs[idx] = llrintf(xs[idx] + diff_x / len);
        ys[idx] = llrintf(ys[idx] + diff_y / len);
    }
}

struct AgentSoaCuda {
    float* xs, *ys, *dest_xs, *dest_ys, *dest_rs;
};

AgentSoaCuda AllocateAgentSoaDevice(std::size_t agent_count) {
    AgentSoaCuda cuda_soa;
    int alloc_size = agent_count * sizeof(float);
    cudaMalloc((void**) &cuda_soa.xs, alloc_size);
    cudaMalloc((void**) &cuda_soa.ys, alloc_size);
    cudaMalloc((void**) &cuda_soa.dest_xs, alloc_size);
    cudaMalloc((void**) &cuda_soa.dest_ys, alloc_size);
    cudaMalloc((void**) &cuda_soa.dest_rs, alloc_size);
    return cuda_soa;
}

void InitDevice(const AgentSoa& cpu_soa, const AgentSoaCuda& gpu_soa) {
    int alloc_size = cpu_soa.size * sizeof(float);
    cudaMemcpy(gpu_soa.xs, cpu_soa.xs, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.ys, cpu_soa.ys, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_xs, cpu_soa.dest_xs, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_ys, cpu_soa.dest_ys, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_rs, cpu_soa.dest_rs, alloc_size, cudaMemcpyHostToDevice);
}

void CopyHostToDevice(const AgentSoa& host_pinned, const AgentSoaCuda& device, std::size_t agent_count) {
    int alloc_size = agent_count * sizeof(float);

    cudaMemcpy(device.dest_xs, host_pinned.dest_xs, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device.dest_ys, host_pinned.dest_ys, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device.dest_rs, host_pinned.dest_rs, alloc_size, cudaMemcpyHostToDevice);
}

void CopyDeviceToHost(const AgentSoa& host_pinned, const AgentSoaCuda& device, std::size_t agent_count) {
    int alloc_size = agent_count * sizeof(float);

    cudaMemcpy(host_pinned.xs, device.xs, alloc_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_pinned.ys, device.ys, alloc_size, cudaMemcpyDeviceToHost);
}

static AgentSoaCuda g_device;

void Model::tickCuda() {
    if (!agent_soa) {
        tickSeq();
        agent_soa = new AgentSoa(agents, AgentSoa::MemType::kPinned);
        
        for(std::size_t i = 0; i != agents.size(); ++i) {
            agents[i]->x_ptr = &agent_soa->xs[i];
            agents[i]->y_ptr = &agent_soa->ys[i];
        }

        g_device = AllocateAgentSoaDevice(agent_soa->size);
        InitDevice(*agent_soa, g_device);
    }

    int size = agent_soa->size;
    agent_soa->ComputeNextDestination();
    CopyHostToDevice(*agent_soa, g_device, size);

    int num_blocks = (agent_soa->size + 1023)/1024;
    updateAgentPos<<<num_blocks, 1024>>>(
        size,
        g_device.xs,
        g_device.ys,
        g_device.dest_xs,
        g_device.dest_ys,
        g_device.dest_rs
    );

    CopyDeviceToHost(*agent_soa, g_device, size);
}

} // namespace Ped
