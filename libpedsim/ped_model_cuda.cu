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

void Model::tickCuda() {
    if (!agent_soa) {
        tickSeq();
        agent_soa = new AgentSoa(agents, AgentSoa::MemType::kPinned);
        
        for(std::size_t i = 0; i != agents.size(); ++i) {
            agents[i]->x_ptr = &agent_soa->xs[i];
            agents[i]->y_ptr = &agent_soa->ys[i];
        }
    }

    agent_soa->ComputeNextDestination();

    int num_blocks = (agent_soa->size + 1023)/1024;
    updateAgentPos<<<num_blocks, 1024>>>(
        agent_soa->size,
        agent_soa->xs,
        agent_soa->ys,
        agent_soa->dest_xs,
        agent_soa->dest_ys,
        agent_soa->dest_rs
    );
}

} // namespace Ped
