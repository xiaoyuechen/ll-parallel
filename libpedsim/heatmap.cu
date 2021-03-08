#include <cuda_runtime.h>
#include <stdio.h>

#include <memory>
#include <algorithm>

#include "ped_model.h"


namespace Ped {

__constant__ int kWeightMatrix[5][5];


int* h_hm;

void Model::SetupHeatmapCuda() {
  cudaMallocHost(&h_hm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  cudaMemset(h_hm, 0, SIZE * SIZE * sizeof(int));
  cudaMalloc(&hm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  cudaMalloc(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  cudaMalloc(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  
  cudaMallocHost(&desired_xs, agents.size() * sizeof(int));
  cudaMallocHost(&desired_ys, agents.size() * sizeof(int));
  
  const int w[5][5] = 
  {{1, 4, 7, 4, 1},
  {4, 16, 26, 16, 4},
  {7, 26, 41, 26, 7},
  {4, 16, 26, 16, 4},
  {1, 4, 7, 4, 1}};
  
  cudaMemcpyToSymbol(kWeightMatrix, w, 5*5*sizeof(int));
  
  cudaMallocHost(&h_bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
  for (int i = 0; i < SCALED_SIZE; i++) {
    blurred_heatmap[i] = h_bhm + SCALED_SIZE * i;
  }
}

__global__ void FadeHeatmap(int* hm, int n) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  hm[y * n + x] = (int)round(hm[y * n + x] * 0.80);
}

__global__ void IntensifyHeat(
  const int* desired_xs, 
  const int* desired_ys, 
  int* hm, 
  int agents_count,
  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < agents_count) {
    int offset = desired_ys[idx] * n + desired_xs[idx];
    atomicAdd(hm + offset, 40);
    if(hm[offset] > 255) {
      hm[offset] = 255;
    }
  }
}

__global__ void ScaleHeatmap(int* hm, int* shm, int ratio, int n) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int val = hm[y * n + x];
  for(int cy = 0; cy < ratio; ++cy)
    for(int cx = 0; cx < ratio; ++cx)
      shm[(y * ratio + cy) * n * ratio + x * ratio + cx] = val;
}

#define WEIGHTSUM 273

__global__ void BlurHeatmap(int* shm, int* bhm, int n) {
  __shared__ int blk [32][32];

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  blk[threadIdx.y][threadIdx.x] = shm[y * n + x];
  __syncthreads();
  if(2 <= x && x < n - 2 && 2 <= y && y < n - 2) {
    int sum = 0;
    for (int k = -2; k < 3; k++) {
      for (int l = -2; l < 3; l++) {
        int cy = threadIdx.y + k;
        int cx = threadIdx.x + l;
        int v;
        if(0 <= cy && cy < 32 && 0 <= cx && cx < 32)
          v = blk[cy][cx];
        else
          v = shm[(y + k) * n + x + l];
        sum += kWeightMatrix[2 + k][2 + l] * v;
      }
    }
    int val = sum / WEIGHTSUM;
    bhm[y * n + x] = 0x00FF0000 | val << 24;
  }

}

  // Weights for blur filter
  constexpr int w[5][5] = {{1, 4, 7, 4, 1},
                       {4, 16, 26, 16, 4},
                       {7, 26, 41, 26, 7},
                       {4, 16, 26, 16, 4},
                       {1, 4, 7, 4, 1}};

void BlurHeatmapCPU(int* shm, int* bhm){ 
  // Apply gaussian blurfilter
  for (int i = 2; i < SCALED_SIZE - 2; i++) {
    for (int j = 2; j < SCALED_SIZE - 2; j++) {
      int sum = 0;
      for (int k = -2; k < 3; k++) {
        for (int l = -2; l < 3; l++) {
          sum += w[2 + k][2 + l] * shm[(i + k) * SCALED_SIZE + j + l];
        }
      }
      int value = sum / WEIGHTSUM;
      bhm[i * SCALED_SIZE + j] = 0x00FF0000 | value << 24;
    }
  }
}

void CreateHeatmapCPU(int* heatmap, AgentSoa* agent_soa) {
  # pragma omp parallel for
  for (int x = 0; x < SIZE; x++) {
    for (int y = 0; y < SIZE; y++) {
      // heat fades
      heatmap[y * SIZE + x] = (int)round(heatmap[y * SIZE + x] * 0.80);
    }
  }

  # pragma omp parallel for
  // Count how many agents want to go to each location
  for (int i = 0; i < agent_soa->size; i++) {
    int x = agent_soa->desired_xs[i];
    int y = agent_soa->desired_ys[i];

    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
      continue;
    }

    #pragma omp atomic update
    heatmap[y * SIZE + x] += 40;
  }
}

void ScaleHeatmapCPU(int* hm, int* shm) {
  #pragma omp parallel for
  for (int y = 0; y < SIZE; y++) {
    for (int x = 0; x < SIZE; x++) {
    int val = hm[y * SIZE + x];
    for(int cy = 0; cy < CELLSIZE; ++cy)
      for(int cx = 0; cx < CELLSIZE; ++cx)
        shm[(y * CELLSIZE + cy) * SCALED_SIZE + x * CELLSIZE + cx] = val;
    }
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
  //////// end of init ////////
  
  auto tick_start = std::chrono::high_resolution_clock::now();

  cudaEvent_t start[3], stop[3];
  for(int i = 0; i != 3; ++i) {
    cudaEventCreate(&start[i]);
    cudaEventCreate(&stop[i]);
  }

  // scale heatmap
  {
    cudaEventRecord(start[1], 0);

    dim3 threads_per_block(32, 32);
    dim3 num_blocks(SIZE / threads_per_block.x, SIZE / threads_per_block.y);
    ScaleHeatmap<<<num_blocks, threads_per_block>>>(hm, shm, CELLSIZE, SIZE);

    cudaEventRecord(stop[1], 0);
  }

  // blur heatmap
  {
    cudaEventRecord(start[2], 0);
    dim3 threads_per_block(32, 32);
    dim3 num_blocks(SCALED_SIZE / threads_per_block.x, SCALED_SIZE / threads_per_block.y);
    BlurHeatmap<<<num_blocks, threads_per_block>>>(shm, bhm, SCALED_SIZE);
    cudaEventRecord(stop[2], 0);
  }


  /////// collision /////////
  ComputeDesiredPos();
  std::transform(agent_soa->desired_xs, agent_soa->desired_xs + agent_soa->size, 
    desired_xs, [](float x){return int(x);});
  std::transform(agent_soa->desired_ys, agent_soa->desired_ys + agent_soa->size, 
    desired_ys, [](float x){return int(x);});
  SortAgents(agent_soa->xs, *agent_idx_array, agents.size());
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

  // CPU heatmap
  CreateHeatmapCPU(h_hm, agent_soa);
  // ScaleHeatmapCPU(hm, h_shm);

  // timing
  auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now() - tick_start);
  float cpu_end = cpu_duration.count() / 1000;

  cudaEventSynchronize(stop[2]);
  float* times[] = {&heatmap_creation_time, &heatmap_scaling_time, &heatmap_blurring_time};
  for(int i = 0; i < 3; ++i) {
    float duration;
    cudaEventElapsedTime(&duration, start[i], stop[i]);
    *times[i] += duration;
  }
  
  float gpu_duration;
  cudaEventElapsedTime(&gpu_duration, start[0], stop[2]);
  float gpu_end = gpu_duration;
  
  imbalance += (gpu_end - cpu_end) / gpu_end;

  cudaStream_t streams[2];
  for (int i = 0; i < 2; i++)
    cudaStreamCreate(&streams[i]);
  cudaMemcpyAsync(h_bhm, bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(hm, h_hm, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice, streams[1]);
  cudaDeviceSynchronize();
}

}  // namespace Ped
