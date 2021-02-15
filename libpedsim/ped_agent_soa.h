#pragma once

#include <smmintrin.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "ped_agent.h"

namespace Ped {

struct AgentSoa {
  static constexpr std::size_t kAlignment = 16;

  AgentSoa() = default;

  explicit AgentSoa(const std::vector<Tagent*>& agents);

  AgentSoa(const AgentSoa&) = delete;

  ~AgentSoa();

  void ComputeNextDestination() noexcept {
    for (int i = 0; i < size; ++i) {
      // compute if agent reached its current destination
      double diffX = dest_xs[i] - xs[i];
      double diffY = dest_ys[i] - ys[i];
      double length = sqrt(diffX * diffX + diffY * diffY);
      bool agentReachedDestination = length < dest_rs[i];

      // If agent has reached destination (or has no current
      // destination); get next destination if available
      if (agentReachedDestination && !waypoints[i]->empty()) {
        current_waypoint_indice[i] =
            (current_waypoint_indice[i] + 1) % waypoints[i]->size();
        dest_xs[i] = (*waypoints[i])[current_waypoint_indice[i]].getx();
        dest_ys[i] = (*waypoints[i])[current_waypoint_indice[i]].gety();
        dest_rs[i] = (*waypoints[i])[current_waypoint_indice[i]].getr();
      }
    }
  }

  std::size_t size;
  float* xs;
  float* ys;
  float* desired_xs;
  float* desired_ys;
  float* dest_xs;
  float* dest_ys;
  float* dest_rs;
  int* current_waypoint_indice;
  const std::vector<Twaypoint>** waypoints;
};

}  // namespace Ped
