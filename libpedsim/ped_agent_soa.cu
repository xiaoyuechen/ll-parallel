#include "ped_agent_soa.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace Ped {

void* MallocAligned(std::size_t bytes) {
  return _mm_malloc(bytes, AgentSoa::kAlignment); 
}

void* MallocPinned(std::size_t bytes) {
  void* result;
  cudaMallocHost(&result, bytes);
  return result;
}

AgentSoa::AgentSoa(const std::vector<Tagent*>& agents, MemType mem_type) {
  this->mem_type = mem_type;

  void* (*allocate_func)(std::size_t);
  if(mem_type==MemType::kAligned) allocate_func = &MallocAligned;
  else allocate_func = &MallocPinned;

  size = agents.size();
  int alloc_size = ((agents.size() + 3) / 4) * 4;

  float** arrs[] = {
      &xs, &ys, &desired_xs, &desired_ys, &dest_xs, &dest_ys, &dest_rs,
  };

  for (auto arr : arrs) {
    *arr = static_cast<float*>(allocate_func(alloc_size * sizeof(float)));
  }

  current_waypoint_indice =
      static_cast<int*>(allocate_func(alloc_size * sizeof(int)));

  waypoints = (const std::vector<Twaypoint>**)allocate_func(alloc_size * sizeof(void*));

  std::transform(agents.begin(), agents.end(), xs,
                 [](const auto& agent) { return agent->x; });

  std::transform(agents.begin(), agents.end(), ys,
                 [](const auto& agent) { return agent->y; });

  std::transform(agents.begin(), agents.end(), desired_xs,
                 [](const auto& agent) { return agent->desiredPositionX; });

  std::transform(agents.begin(), agents.end(), desired_ys,
                 [](const auto& agent) { return agent->desiredPositionY; });

  std::transform(
      agents.begin(), agents.end(), current_waypoint_indice,
      [](const auto& agent) { return agent->current_waypoint_pointer; });

  std::transform(agents.begin(), agents.end(), dest_xs,
                 [](const auto& agent) { return agent->destination->getx(); });

  std::transform(agents.begin(), agents.end(), dest_ys,
                 [](const auto& agent) { return agent->destination->gety(); });

  std::transform(agents.begin(), agents.end(), dest_rs,
                 [](const auto& agent) { return agent->destination->getr(); });

  std::transform(agents.begin(), agents.end(), waypoints,
                 [](const auto& agent) { return &(agent->waypoints); });
}

void FreePinned(void* mem) {
  cudaFreeHost(mem);
}

AgentSoa::~AgentSoa() {
  if(mem_type == MemType::kNone)
    return;

  void (*free)(void*);
  if(mem_type == MemType::kAligned)
    free = &_mm_free;
  else if (mem_type == MemType::kPinned)
    free = &FreePinned;

  float* arrs[] = {xs, ys, desired_xs, desired_ys};
  for (auto arr : arrs) {
    free(arr);
  }

  free(current_waypoint_indice);

  float* darrs[] = {
      dest_xs,
      dest_ys,
      dest_rs,
  };

  for (auto darr : darrs) {
    free(darr);
  }

  free(waypoints);
}

}  // namespace Ped