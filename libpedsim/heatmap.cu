#include <memory>
#include <cuda_runtime.h>
#include "ped_model.h"
#include <stdio.h>

#define BLOCK_NUMBER 4
#define BLOCK_WIDTH 256

namespace Ped {
  __global__ void InitSBHeatmap(int* bhm, int* shm, int** scaled_heatmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    scaled_heatmap[tid] = shm + SCALED_SIZE * tid;
  }

  void Model::setupHeatmapCuda() {
    // cudaStream_t s[6];
    // for(int i = 0; i != 6; ++i) {
    //     cudaStreamCreate(s + i);
    // }

    int *hm, *shm, *bhm;
    
    cudaMalloc(&hm, SIZE * SIZE * sizeof(int));
    cudaMalloc(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc(&heatmap, SIZE * sizeof(int*));
    cudaMalloc(&scaled_heatmap, SCALED_SIZE * sizeof(int*));

    //we need to calculate them on GPU as well?
    cudaMalloc(&desired_xs, 256 * sizeof(int));
    cudaMalloc(&desired_ys, 256 * sizeof(int));

    cudaMallocHost(&blurred_heatmap, SCALED_SIZE * sizeof(int*));
    cudaMemset(hm, 0, SIZE * SIZE);
    cudaMemset(shm, 0, SCALED_SIZE * SCALED_SIZE);
    cudaMemset(bhm, 0, SCALED_SIZE * SCALED_SIZE);

    InitSBHeatmap<<<CELLSIZE,SIZE>>>(bhm, shm, scaled_heatmap);
    cudaDeviceSynchronize();
  }


  __global__ void heatFades(int* heatmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int row = 0; row < SIZE; row++) {
      heatmap[row * SIZE + tid] = (int)round(heatmap[row * SIZE + tid] * 0.80);
    }
  }

  __global__ void coloringTheMap(int* heatmap, const int agents, int* desired_xs, int* desired_ys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid > agents) return;
    if(desired_xs[tid]<0||desired_xs[tid]>=SIZE||desired_ys[tid]<0||desired_xs[tid]>=SIZE) return;

    atomicAdd(heatmap + desired_ys[tid] * SIZE + desired_xs[tid], 40);
	  atomicMin(heatmap + desired_ys[tid] * SIZE + desired_xs[tid], 255);
  }

  void Model::updateHeatmapCuda() {
    float time1, time2, time3;
    cudaEvent_t fade_start, fade_stop;
    cudaEventCreate(&fade_start);
    cudaEventCreate(&fade_stop);
    cudaEventRecord(fade_start, 0);
  
    heatFades<<<1, SIZE>>>(*heatmap);
    
    cudaEventRecord(fade_stop, 0);
    cudaEventSynchronize(fade_stop);
    cudaEventElapsedTime(&time1, fade_start, fade_stop);
    cudaEventDestroy(fade_start);
    cudaEventDestroy(fade_stop);

    cudaEvent_t coloring_start, coloring_stop;
    cudaEventCreate(&coloring_start);
    cudaEventCreate(&coloring_stop);
    cudaEventRecord(coloring_start, 0);
  
    coloringTheMap<<<1, SIZE>>>(*heatmap, agents.size(), desired_xs, desired_ys);
    
    cudaEventRecord(coloring_stop, 0);
    cudaEventSynchronize(coloring_stop);
    cudaEventElapsedTime(&time2, coloring_start, coloring_stop);
    cudaEventDestroy(coloring_start);
    cudaEventDestroy(coloring_stop);

    //need another gaussian filtering cuda kernel
  }

}  // namespace Ped
