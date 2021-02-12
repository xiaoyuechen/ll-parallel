//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"

#include <smmintrin.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stack>
#include <thread>

#include "cuda_testkernel.h"
#include "ped_model.h"
#include "ped_waypoint.h"

namespace Ped {

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario,
                       std::vector<Twaypoint*> destinationsInScenario,
                       IMPLEMENTATION implementation) {
  // Convenience test: does CUDA work on this machine?
  cuda_test();

  // Set
  agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(),
                                     agentsInScenario.end());

  // Set up destinations
  destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(),
                                              destinationsInScenario.end());

  // Sets the chosen implemenation. Standard in the given code is SEQ
  this->implementation = implementation;

  // Set up heatmap (relevant for Assignment 4)
  setupHeatmapSeq();
}

namespace {

void move_agent(Ped::Tagent& agent) {
  agent.computeNextDesiredPosition();
  agent.setX(agent.getDesiredX());
  agent.setY(agent.getDesiredY());
}

void thread_move_agents(std::vector<Ped::Tagent*>::iterator begin,
                        std::vector<Ped::Tagent*>::iterator end) {
  for (auto agent = begin; agent < end; ++agent) {
    move_agent(**agent);
  }
}

// __global__ void

}  // namespace

void Ped::Model::tickSeq() {
  for (auto agent : agents) {
    move_agent(*agent);
  }
}

void Ped::Model::tickOmp() {
#pragma omp parallel for
  for (auto it = agents.begin(); it < agents.end(); it++) {
    move_agent(**it);
  }
}

void Ped::Model::tickThread() {
  static constexpr std::size_t kMaxThreads = 8;
  auto thread_array = std::array<std::thread, kMaxThreads>{};
  std::size_t thread_count =
      std::min((std::size_t)omp_get_max_threads(), kMaxThreads);
  std::size_t chunk = (std::size_t)ceil((double)agents.size() / thread_count);

  for (std::size_t i = 0; i < thread_count; i++) {
    auto begin = agents.begin() + i * chunk;
    auto end = begin + chunk;
    if (end > agents.end()) end = agents.end();
    thread_array[i] = std::thread(thread_move_agents, begin, end);
  }

  for (std::size_t i = 0; i < thread_count; i++) {
    thread_array[i].join();
  }
}

void PrintAgentSoa(const AgentSoa& soa, int idx) {
  printf("xy: [%f, %f]; dest_xyr: [%f, %f, %f], idx: %d\n", soa.xs[idx],
         soa.ys[idx], soa.dest_xs[idx], soa.dest_ys[idx], soa.dest_rs[idx],
         soa.current_waypoint_indice[idx]);
}

void PrintAgent(const Tagent& agent) {
  printf("xy: [%f, %f]; dest_xyr: [%f, %f, %f], idx: %d\n", float(agent.x),
         float(agent.y), agent.destination->getx(), agent.destination->gety(),
         agent.destination->getr(), agent.current_waypoint_pointer);
}

void Ped::Model::tickVector() {
  if (!agent_soa) {
    tickSeq();
    agent_soa = new AgentSoa(agents);
  }

  agent_soa->ComputeNextDestination();

  // #pragma omp parallel for
  int iter = (int)::ceilf((float)agent_soa->size / 4);
  for (int i = 0; i < iter; ++i) {
    int stride = i * 4;

    // SIMD code

    // for loop seq next waypoint

    // SIMD code
    __m128 dest_x = _mm_load_ps(&agent_soa->dest_xs[stride]);
    __m128 dest_y = _mm_load_ps(&agent_soa->dest_ys[stride]);
    __m128 x = _mm_load_ps(&agent_soa->xs[stride]);
    __m128 y = _mm_load_ps(&agent_soa->ys[stride]);

    __m128 diff_x = _mm_sub_ps(dest_x, x);
    __m128 diff_y = _mm_sub_ps(dest_y, y);
    __m128 len = _mm_sqrt_ps(
        _mm_add_ps(_mm_mul_ps(diff_x, diff_x), _mm_mul_ps(diff_y, diff_y)));
    __m128 desired_x = _mm_add_ps(x, _mm_div_ps(diff_x, len));
    __m128 desired_y = _mm_add_ps(y, _mm_div_ps(diff_y, len));

    desired_x = _mm_round_ps(desired_x,
                             (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    desired_y = _mm_round_ps(desired_y,
                             (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    // _mm_store_ps(agent_soa->desired_xs + stride, desired_x);
    // _mm_store_ps(agent_soa->desired_ys + stride, desired_y);
    _mm_store_ps(&agent_soa->xs[stride], desired_x);
    _mm_store_ps(&agent_soa->ys[stride], desired_y);
  }

#pragma omp parallel for
  for (int i = 0; i < agent_soa->size; ++i) {
    agents[i]->setX(agent_soa->xs[i]);
    agents[i]->setY(agent_soa->ys[i]);
  }
}

void Ped::Model::tick() {
  switch (implementation) {
    case IMPLEMENTATION::SEQ:
      tickSeq();
      break;
    case IMPLEMENTATION::PTHREAD:
      tickThread();
      break;
    case IMPLEMENTATION::OMP:
      tickOmp();
      break;
    case IMPLEMENTATION::VECTOR:
      tickVector();
      break;
  }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent* agent) {
  // Search for neighboring agents
  set<const Ped::Tagent*> neighbors =
      getNeighbors(agent->getX(), agent->getY(), 2);

  // Retrieve their positions
  std::vector<std::pair<int, int>> takenPositions;
  for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin();
       neighborIt != neighbors.end(); ++neighborIt) {
    std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
    takenPositions.push_back(position);
  }

  // Compute the three alternative positions that would bring the agent
  // closer to his desiredPosition, starting with the desiredPosition itself
  std::vector<std::pair<int, int>> prioritizedAlternatives;
  std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
  prioritizedAlternatives.push_back(pDesired);

  int diffX = pDesired.first - agent->getX();
  int diffY = pDesired.second - agent->getY();
  std::pair<int, int> p1, p2;
  if (diffX == 0 || diffY == 0) {
    // Agent wants to walk straight to North, South, West or East
    p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
    p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
  } else {
    // Agent wants to walk diagonally
    p1 = std::make_pair(pDesired.first,

                        agent->getY());
    p2 = std::make_pair(agent->getX(), pDesired.second);
  }
  prioritizedAlternatives.push_back(p1);
  prioritizedAlternatives.push_back(p2);

  // Find the first empty alternative position
  for (std::vector<pair<int, int>>::iterator it =
           prioritizedAlternatives.begin();
       it != prioritizedAlternatives.end(); ++it) {
    // If the current position is not yet taken by any neighbor
    if (std::find(takenPositions.begin(), takenPositions.end(), *it) ==
        takenPositions.end()) {
      // Set the agent's position
      agent->setX((*it).first);
      agent->setY((*it).second);

      break;
    }
  }
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents
/// (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
  // create the output list
  // ( It would be better to include only the agents close by, but this
  // programmer is lazy.)
  return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
  // Nothing to do here right now.
}

Ped::Model::~Model() {
  std::for_each(agents.begin(), agents.end(),
                [](Ped::Tagent* agent) { delete agent; });
  std::for_each(destinations.begin(), destinations.end(),
                [](Ped::Twaypoint* destination) { delete destination; });

  delete agent_soa;
}

}  // namespace Ped
