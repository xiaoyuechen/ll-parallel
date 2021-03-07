#include <cuda_runtime.h>
#include <stdio.h>

#include <memory>
#include <algorithm>

#include "ped_model.h"


namespace Ped {

__constant__ int kWeightMatrix[5][5];


void Model::SetupHeatmapCuda() {
  cudaMalloc(&hm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  cudaMemset(hm, 0, SIZE * SIZE * sizeof(int));
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


__global__ void BlurHeatmap(int* shm, int* bhm, int n) {
  #define WEIGHTSUM 273
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



void Model::UpdateHeatmapCuda() {
  {
    dim3 threads_per_block(32, 32);
    dim3 num_blocks(SIZE / threads_per_block.x, SIZE / threads_per_block.y);
    FadeHeatmap<<<num_blocks, threads_per_block>>>(hm, SIZE);
  }
  
  {
    std::transform(agent_soa->desired_xs, agent_soa->desired_xs + agent_soa->size, 
      desired_xs, [](float x){return int(x);});
    std::transform(agent_soa->desired_ys, agent_soa->desired_ys + agent_soa->size, 
      desired_ys, [](float x){return int(x);});
    int threads_per_block = 1024;
    int num_blocks = (agent_soa->size + threads_per_block - 1) / threads_per_block;
    IntensifyHeat<<<num_blocks, threads_per_block>>>(desired_xs, desired_ys, hm, agent_soa->size, SIZE);
  }

  {
    dim3 threads_per_block(32, 32);
    dim3 num_blocks(SIZE / threads_per_block.x, SIZE / threads_per_block.y);
    ScaleHeatmap<<<num_blocks, threads_per_block>>>(hm, shm, CELLSIZE, SIZE);
  }

  {
    dim3 threads_per_block(32, 32);
    dim3 num_blocks(SCALED_SIZE / threads_per_block.x, SCALED_SIZE / threads_per_block.y);
    BlurHeatmap<<<num_blocks, threads_per_block>>>(shm, bhm, SCALED_SIZE);
  }

  {
    cudaMemcpy(h_bhm, bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  }

}

}  // namespace Ped
