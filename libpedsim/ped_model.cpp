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
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
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
  switch (implementation) {
    case Ped::IMPLEMENTATION::SEQ:
      setupHeatmapSeq();
      break;
    default:
      setupHeatmapCuda();
      break;
  }
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

}  // namespace

void Ped::Model::tickSeq() {
  for (auto agent : agents) {
    agent->computeNextDesiredPosition();
    move(agent);
  }
  updateHeatmapSeq();
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
    agent_soa = new AgentSoa(agents, AgentSoa::MemType::kAligned);
    for (std::size_t i = 0; i != agents.size(); ++i) {
      agents[i]->x_ptr = &agent_soa->xs[i];
      agents[i]->y_ptr = &agent_soa->ys[i];
    }
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
}

void Model::ComputeDesiredPos() {
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
    _mm_store_ps(&agent_soa->desired_xs[stride], desired_x);
    _mm_store_ps(&agent_soa->desired_ys[stride], desired_y);
  }
}

std::uint32_t& cell(State& state, int x, int y) {
  return state.state[x + state.offset_x][y + state.offset_y];
}

static constexpr std::size_t kStateX = 300;
static constexpr std::size_t kStateY = 200;

void Model::tickRegion() {
  if (!agent_soa) {
    for (auto agent : agents) {
    agent->computeNextDesiredPosition();
    move(agent);
    }
    agent_soa = new AgentSoa(agents, AgentSoa::MemType::kAligned);
    for (std::size_t i = 0; i != agents.size(); ++i) {
      agents[i]->x_ptr = &agent_soa->xs[i];
      agents[i]->y_ptr = &agent_soa->ys[i];
    }
    agent_idx_array = new AgentIdxArray(agents.size());

    state.offset_x = 50;
    state.offset_y = 0;
    state.state = new std::uint32_t*[kStateX];
    for (int i = 0; i < kStateX; ++i)
      state.state[i] = new std::uint32_t[kStateY];

    for (int i = 0; i != kStateX; ++i)
      for (int j = 0; j != kStateY; ++j) state.state[i][j] = ~std::uint32_t(0);

    for (std::size_t i = 0; i != agents.size(); ++i) {
      int x = (int)agent_soa->xs[i];
      int y = (int)agent_soa->ys[i];
      cell(state, x, y) = i;
    }
  }
  SortAgents(agent_soa->xs, *agent_idx_array, agents.size());
  ComputeDesiredPos();
  updateHeatmapCuda();

#pragma omp parallel
  {
    std::size_t region_agent_count =
        (std::size_t)ceil((double)agents.size() / omp_get_num_threads());
    int thread_id = omp_get_thread_num();
    std::uint32_t* begin =
        agent_idx_array->indice + thread_id * region_agent_count;
    std::uint32_t* end =
        agent_idx_array->indice + (thread_id + 1) * region_agent_count;
    if (end > agent_idx_array->indice + agent_soa->size)
      end = agent_idx_array->indice + agent_soa->size;
    // printf("%u, %u\n", *begin, *end);
    move(begin, end);
  }
}

std::array<std::pair<int, int>, 3> get_desired_moves(int x, int y,
                                                     int desired_x,
                                                     int desired_y) noexcept {
  auto result = std::array<std::pair<int, int>, 3>{};
  result[0] = std::make_pair(desired_x, desired_y);
  auto diff_x = desired_x - x;
  auto diff_y = desired_y - y;
  if (diff_x == 0 || diff_y == 0) {
    // Agent wants to walk straight to North, South, West or East
    result[1] = std::make_pair(desired_x + diff_y, desired_y + diff_x);
    result[2] = std::make_pair(desired_x - diff_y, desired_y - diff_x);
  } else {
    // Agent wants to walk diagonally
    result[1] = std::make_pair(desired_x, y);
    result[2] = std::make_pair(x, desired_y);
  }
  return result;
}

void Model::move(std::uint32_t* begin, std::uint32_t* end) {
  static std::random_device rd;
  static std::mt19937 g(rd());
  float region_begin = agent_soa->xs[*begin];
  float region_end;
  if (end == agent_idx_array->indice + agent_soa->size)
    region_end = agent_soa->xs[*(end - 1)];
  else
    region_end = agent_soa->xs[*end];

  std::shuffle(begin, end, g);
  for (auto iter = begin; iter != end; ++iter) {
    std::uint32_t agent_idx = *iter;
    int x = (int)agent_soa->xs[agent_idx];
    int y = (int)agent_soa->ys[agent_idx];
    int desired_x = (int)agent_soa->desired_xs[agent_idx];
    int desired_y = (int)agent_soa->desired_ys[agent_idx];
    auto desired_moves = get_desired_moves(x, y, desired_x, desired_y);

    for (auto move : desired_moves) {
      int move_x = move.first;
      int move_y = move.second;

      bool local_move = move_x > region_begin && move_x < region_end;
      if (local_move) {
        if (cell(state, move_x, move_y) == ~std::uint32_t(0)) {
          cell(state, move_x, move_y) = agent_idx;
          cell(state, x, y) = ~std::uint32_t(0);
          agent_soa->xs[agent_idx] = move_x;
          agent_soa->ys[agent_idx] = move_y;
          break;
        }
      } else {
        if (__sync_bool_compare_and_swap(&cell(state, move_x, move_y),
                                         ~std::uint32_t(0), agent_idx)) {
          cell(state, x, y) = ~std::uint32_t(0);
          agent_soa->xs[agent_idx] = move_x;
          agent_soa->ys[agent_idx] = move_y;
          break;
        }
      }
    }
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
    case IMPLEMENTATION::CUDA:
      tickCuda();
      break;
    case IMPLEMENTATION::REGION:
      tickRegion();
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
    p1 = std::make_pair(pDesired.first, agent->getY());
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
