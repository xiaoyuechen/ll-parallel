#include "ped_agent_soa.h"

namespace Ped {

AgentSoa::AgentSoa(const std::vector<Tagent*>& agents) {
  size = agents.size();
  int* arrs[] = {xs, ys, desired_xs, desired_ys, current_waypoint_indice};
  for (auto& arr : arrs) {
    arr = static_cast<int*>(_mm_malloc(size * sizeof(int), kAlignment));
  }

  double* darrs[] = {
      dest_xs,
      dest_ys,
      dest_rs,
  };

  for (auto& darr : darrs) {
    darr = static_cast<double*>(_mm_malloc(size * sizeof(double), kAlignment));
  }

  waypoints = (const std::vector<Twaypoint>**)_mm_malloc(size * sizeof(void*),
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
                 [](const auto& agent) { return agent->destination->getx(); });

  std::transform(agents.begin(), agents.end(), dest_rs,
                 [](const auto& agent) { return agent->destination->getr(); });

  std::transform(agents.begin(), agents.end(), waypoints,
                 [](const auto& agent) { return &(agent->waypoints); });
}

}  // namespace Ped
