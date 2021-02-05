//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <emmintrin.h>
#include <omp.h>

#include <algorithm>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "ped_agent.h"

namespace Ped {

// The implementation modes for Assignment 1 + 2:
// chooses which implementation to use for tick()
enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

struct AgentsData {
  explicit AgentsData(const std::vector<Tagent*>& agents) {
    size = agents.size();
    int* arrs[] = {xs, ys, desired_xs, desired_ys, current_way_point_idice};
    for (auto& arr : arrs) {
      arr = static_cast<int*>(_mm_malloc(size * sizeof(int), 16));
    }

    double* darrs[] = {
        dest_xs,
        dest_ys,
        dest_rs,
    };

    for (auto& darr : darrs) {
      darr = static_cast<double*>(_mm_malloc(size * sizeof(double), 16));
    }

    waypoints =
        (const std::vector<Twaypoint>**)_mm_malloc(size * sizeof(void*), 16);

    std::transform(agents.begin(), agents.end(), xs,
                   [](const auto& agent) { return agent->x; });

    std::transform(agents.begin(), agents.end(), ys,
                   [](const auto& agent) { return agent->y; });

    std::transform(agents.begin(), agents.end(), desired_xs,
                   [](const auto& agent) { return agent->desiredPositionX; });

    std::transform(agents.begin(), agents.end(), desired_ys,
                   [](const auto& agent) { return agent->desiredPositionY; });

    std::transform(
        agents.begin(), agents.end(), current_way_point_idice,
        [](const auto& agent) { return agent->current_waypoint_pointer; });

    std::transform(
        agents.begin(), agents.end(), dest_xs,
        [](const auto& agent) { return agent->destination->getx(); });

    std::transform(
        agents.begin(), agents.end(), dest_ys,
        [](const auto& agent) { return agent->destination->getx(); });

    std::transform(
        agents.begin(), agents.end(), dest_rs,
        [](const auto& agent) { return agent->destination->getr(); });

    std::transform(agents.begin(), agents.end(), waypoints,
                   [](const auto& agent) { return &(agent->waypoints); });
  }

  AgentsData(const AgentsData&) = delete;

  ~AgentsData() {
    int* arrs[] = {xs, ys, desired_xs, desired_ys, current_way_point_idice};
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
    // #pragma omp parallel for
    //     for (int i = 0; i < size; ++i) {
    //       Ped::Twaypoint* nextDestination = NULL;

    //       // compute if agent reached its current destination
    //       double diffX = dest_xs[i] - xs[i];
    //       double diffY = dest_ys[i] - ys[i];
    //       double length = diffX * diffX + diffY * diffY;
    //       bool agentReachedDestination = length < (dest_rs[i] * dest_rs[i]);

    //       if ((agentReachedDestination) && !waypoints.empty()) {
    //         // Case 1: agent has reached destination (or has no current
    //         // destination); get next destination if available
    //         current_waypoint_pointer =
    //             (current_waypoint_pointer + 1) % waypoints.size();
    //         nextDestination = &waypoints[current_waypoint_pointer];
    //       } else {
    //         // Case 2: agent has not yet reached destination, continue to
    //         move
    //         // towards current destination
    //         nextDestination = destination;
    //       }

    //       return nextDestination;
    //     }
  }

  std::size_t size;
  int* xs;
  int* ys;
  int* desired_xs;
  int* desired_ys;
  int* current_way_point_idice;
  double* dest_xs;
  double* dest_ys;
  double* dest_rs;
  const std::vector<Twaypoint>** waypoints;
};

class Model {
 public:
  // Sets everything up
  void setup(std::vector<Tagent*> agentsInScenario,
             std::vector<Twaypoint*> destinationsInScenario,
             IMPLEMENTATION implementation);

  // Coordinates a time step in the scenario: move all agents by one step (if
  // applicable).
  void tick();

  // Returns the agents of this scenario
  const std::vector<Tagent*> getAgents() const { return agents; };

  // Adds an agent to the tree structure
  void placeAgent(const Ped::Tagent* a);

  // Cleans up the tree and restructures it. Worth calling every now and then.
  void cleanup();
  ~Model();

  // Returns the heatmap visualizing the density of agents
  int const* const* getHeatmap() const { return blurred_heatmap; };
  int getHeatmapSize() const;

 private:
  void tickSeq();
  void tickOmp();
  void tickThread();

  // Denotes which implementation (sequential, parallel implementations..)
  // should be used for calculating the desired positions of
  // agents (Assignment 1)
  IMPLEMENTATION implementation;

  // The agents in this scenario
  std::vector<Tagent*> agents;

  // The waypoints in this scenario
  std::vector<Twaypoint*> destinations;

  // Moves an agent towards its next position
  void move(Ped::Tagent* agent);

  ////////////
  /// Everything below here won't be relevant until Assignment 3
  ///////////////////////////////////////////////

  // Returns the set of neighboring agents for the specified position
  set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

  ////////////
  /// Everything below here won't be relevant until Assignment 4
  ///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE* CELLSIZE

  // The heatmap representing the density of agents
  int** heatmap;

  // The scaled heatmap that fits to the view
  int** scaled_heatmap;

  // The final heatmap: blurred and scaled to fit the view
  int** blurred_heatmap;

  void setupHeatmapSeq();
  void updateHeatmapSeq();
};
}  // namespace Ped
#endif
