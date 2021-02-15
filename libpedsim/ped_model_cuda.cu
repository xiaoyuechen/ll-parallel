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

AgentSoaCuda AllocateAgentSoaCuda(std::size_t agent_count) {
    AgentSoaCuda cuda_soa;
    int alloc_size = agent_count * sizeof(float);
    cudaMalloc((void**) &cuda_soa.xs, alloc_size);
    cudaMalloc((void**) &cuda_soa.ys, alloc_size);
    cudaMalloc((void**) &cuda_soa.dest_xs, alloc_size);
    cudaMalloc((void**) &cuda_soa.dest_ys, alloc_size);
    cudaMalloc((void**) &cuda_soa.dest_rs, alloc_size);
    return cuda_soa;
}

void CopyAgentsToCuda(const AgentSoa& cpu_soa, const AgentSoaCuda& gpu_soa) {
    int alloc_size = cpu_soa.size * sizeof(float);
    cudaMemcpy(gpu_soa.xs, cpu_soa.xs, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.ys, cpu_soa.ys, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_xs, cpu_soa.dest_xs, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_ys, cpu_soa.dest_ys, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_rs, cpu_soa.dest_rs, alloc_size, cudaMemcpyHostToDevice);
}

void CopyAgentsDestToCuda(const AgentSoa& cpu_soa, const AgentSoaCuda& gpu_soa) {
    int alloc_size = cpu_soa.size * sizeof(float);
    cudaMemcpy(gpu_soa.dest_xs, cpu_soa.dest_xs, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_ys, cpu_soa.dest_ys, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_soa.dest_rs, cpu_soa.dest_rs, alloc_size, cudaMemcpyHostToDevice);
}

void CopyAgentsPosToHost(const AgentSoa& cpu_soa, const AgentSoaCuda& gpu_soa) {
    int alloc_size = cpu_soa.size * sizeof(float);
    cudaMemcpy(cpu_soa.xs, gpu_soa.xs, alloc_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_soa.ys, gpu_soa.ys, alloc_size, cudaMemcpyDeviceToHost);
}

static AgentSoaCuda g_cuda_soa;

void Model::tickCuda() {
    if (!agent_soa) {
        tickSeq();
        agent_soa = new AgentSoa(agents);

        g_cuda_soa = AllocateAgentSoaCuda(agent_soa->size);
        CopyAgentsToCuda(*agent_soa, g_cuda_soa);
    }

    agent_soa->ComputeNextDestination();
    CopyAgentsDestToCuda(*agent_soa, g_cuda_soa);

    int num_blocks = (agent_soa->size + 1023)/1024;
    updateAgentPos<<<num_blocks, 1024>>>(
        agent_soa->size,
        g_cuda_soa.xs,
        g_cuda_soa.ys,
        g_cuda_soa.dest_xs,
        g_cuda_soa.dest_ys,
        g_cuda_soa.dest_rs
    );

    CopyAgentsPosToHost(*agent_soa, g_cuda_soa);

    for (int i = 0; i < agent_soa->size; ++i) {
        agents[i]->setX(agent_soa->xs[i]);
        agents[i]->setY(agent_soa->ys[i]);
    }
}

} // namespace Ped
