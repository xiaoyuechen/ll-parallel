#include "ped_agent_soa.h"

namespace Ped {

AgentSoa::AgentSoa(const std::vector<Tagent*>& agents) {
  size = agents.size();
  int alloc_size = ((agents.size() + 3) / 4) * 4;

  float** arrs[] = {
      &xs, &ys, &desired_xs, &desired_ys, &dest_xs, &dest_ys, &dest_rs,
  };

  for (auto arr : arrs) {
    *arr = static_cast<float*>(_mm_malloc(alloc_size * sizeof(float), kAlignment));
  }

  current_waypoint_indice =
      static_cast<int*>(_mm_malloc(alloc_size * sizeof(int), kAlignment));

  waypoints = (const std::vector<Twaypoint>**)_mm_malloc(alloc_size * sizeof(void*),
                                                         kAlignment);

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

AgentSoa::~AgentSoa() {
  float* arrs[] = {xs, ys, desired_xs, desired_ys};
  for (auto arr : arrs) {
    _mm_free(arr);
  }

  _mm_free(current_waypoint_indice);

  float* darrs[] = {
      dest_xs,
      dest_ys,
      dest_rs,
  };

  for (auto darr : darrs) {
    _mm_free(darr);
  }

  _mm_free(waypoints);
}

}  // namespace Ped