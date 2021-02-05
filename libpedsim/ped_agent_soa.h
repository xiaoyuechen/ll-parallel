#pragma once

#include <emmintrin.h>

#include <algorithm>
#include <vector>

#include "ped_agent.h"

namespace Ped {

struct AgentSoa {
  static constexpr std::size_t kAlignment = 16;

  explicit AgentSoa(const std::vector<Tagent*>& agents);

  AgentSoa(const AgentSoa&) = delete;

  ~AgentSoa() {
    int* arrs[] = {xs, ys, desired_xs, desired_ys, current_waypoint_indice};
    for (auto arr : arrs) {
      _mm_free(arr);
    }

    double* darrs[] = {
        dest_xs,
        dest_ys,
        dest_rs,
    };

    for (auto darr : darrs) {
      _mm_free(darr);
    }

    _mm_free(waypoints);
  }

  void ComputeNextDestination() noexcept {
    for (int i = 0; i < size; ++i) {
      // compute if agent reached its current destination
      double diffX = dest_xs[i] - xs[i];
      double diffY = dest_ys[i] - ys[i];
      double length = diffX * diffX + diffY * diffY;
      bool agentReachedDestination = length < (dest_rs[i] * dest_rs[i]);

      // If agent has reached destination (or has no current
      // destination); get next destination if available
      if (agentReachedDestination && !waypoints[i]->empty()) {
        current_waypoint_indice[i] =
            (current_waypoint_indice[i] + 1) % (*waypoints[i]).size();
        dest_xs[i] = (*waypoints[i])[current_waypoint_indice[i]].getx();
        dest_ys[i] = (*waypoints[i])[current_waypoint_indice[i]].gety();
        dest_rs[i] = (*waypoints[i])[current_waypoint_indice[i]].getr();
      }
    }
  }

  std::size_t size;
  int* xs;
  int* ys;
  int* desired_xs;
  int* desired_ys;
  int* current_waypoint_indice;
  double* dest_xs;
  double* dest_ys;
  double* dest_rs;
  const std::vector<Twaypoint>** waypoints;
};

}  // namespace Ped
