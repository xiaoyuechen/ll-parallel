#include <memory>
#include <cuda_runtime.h>
#include "ped_model.h"
#include <stdio.h>


#define BLOCK_NUMBER 4
#define BLOCK_WIDTH 256

namespace Ped {
  __global__ void InitSHeatmap(int* shm, int** scaled_heatmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    scaled_heatmap[tid] = shm + SCALED_SIZE * tid;
  }

  __global__ void InitHeatmap(int* hm, int** heatmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    heatmap[tid] = hm + SIZE * tid;
  }

  __global__ void CudaPrint(int* hm) {
    printf("%d\n",hm[255]);
  }

  void Model::setupHeatmapCuda() {
    // cudaStream_t s[6];
    // for(int i = 0; i != 6; ++i) {
    //     cudaStreamCreate(s + i);
    // }

    int *hm, *shm, *bhm;

    cudaMalloc(&hm, SIZE * SIZE * sizeof(int));
    cudaMalloc(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc(&heatmap, SIZE * sizeof(int*));
    cudaMalloc(&scaled_heatmap, SCALED_SIZE * sizeof(int*));

    //we need to calculate them on GPU as well?
    cudaMalloc(&desired_xs, agents.size() * sizeof(int));
    cudaMalloc(&desired_ys, agents.size() * sizeof(int));

    cudaMallocHost(&blurred_heatmap, SCALED_SIZE * sizeof(int*));
    cudaMallocHost(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMemset(hm, 0, SIZE * SIZE);
    cudaMemset(shm, 0, SCALED_SIZE * SCALED_SIZE);
    cudaMemset(bhm, 1, SCALED_SIZE * SCALED_SIZE);


    InitHeatmap<<<1,SIZE>>>(hm, heatmap);
    cudaDeviceSynchronize();

    InitSHeatmap<<<CELLSIZE,SIZE>>>(shm, scaled_heatmap);
    cudaDeviceSynchronize();

    for (int i = 0; i < SCALED_SIZE; i++) {
      blurred_heatmap[i] = bhm + SCALED_SIZE * i;
    }
  }


  __global__ void heatFades(int** heatmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int row = 0; row < SIZE; row++) {
      heatmap[row][tid] = (int)round(heatmap[row][tid] * 0.80);
    }
  }

  __global__ void coloringTheMap(int** heatmap, const int agents, int* desired_xs, int* desired_ys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid > agents) return;
    if(desired_xs[tid]<0 || desired_xs[tid]>SIZE || 
        desired_ys[tid]<0 || desired_ys[tid]>SIZE) 
        return;

    int i = desired_ys[tid];
    atomicAdd(&heatmap[desired_xs[tid]][i], 40);
  }

  __global__ void coloringTheMap1(int** heatmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < SIZE; ++i)
      atomicMin(&heatmap[tid][i], 255);
  }

  __global__ void scalingTheMap(int** heatmap, int** scaled_heatmap){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int x = 0; x < SIZE; x++) {
      int value = heatmap[tid][x];
      for (int cellY = 0; cellY < CELLSIZE; cellY++) {
        for (int cellX = 0; cellX < CELLSIZE; cellX++) {
          scaled_heatmap[tid * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
        }
      }
    }
  }

  __global__ void filteringTheMap(int** scaled_heatmap, int** blurred_heatmap, const int w[5][5]){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Apply gaussian blurfilter
    if((tid > 1) && (tid < SCALED_SIZE - 2))
      for (int j = 2; j < SCALED_SIZE - 2; j++) {
        int sum = 0;
        for (int k = -2; k < 3; k++) {
          for (int l = -2; l < 3; l++) {
            sum += w[2 + k][2 + l] * scaled_heatmap[tid + k][j + l];
          }
        }
        int value = sum / 273;
        // printf("%d\n", value);
        auto temp = 0x00FF0000 | value << 24;
        // blurred_heatmap[tid][j] = temp;
      }

  }

  void Model::updateHeatmapCuda() {
    float time1, time2, time3, time4;
    cudaEvent_t fade_start, fade_stop;
    cudaEventCreate(&fade_start);
    cudaEventCreate(&fade_stop);
    cudaEventRecord(fade_start, 0);

    heatFades<<<1, SIZE>>>(heatmap);

    cudaEventRecord(fade_stop, 0);
    cudaEventSynchronize(fade_stop);
    cudaEventElapsedTime(&time1, fade_start, fade_stop);
    cudaEventDestroy(fade_start);
    cudaEventDestroy(fade_stop);

    // ////////////////////////////////////////////////////
    cudaMemcpyAsync(desired_xs, (*agent_soa).xs, agents.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(desired_ys, (*agent_soa).ys, agents.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t coloring_start, coloring_stop;
    cudaEventCreate(&coloring_start);
    cudaEventCreate(&coloring_stop);
    cudaEventRecord(coloring_start, 0);

    coloringTheMap<<<CELLSIZE, SIZE>>>(heatmap, agents.size(), desired_xs, desired_ys);

    cudaEventRecord(coloring_stop, 0);
    cudaEventSynchronize(coloring_stop);
    cudaEventElapsedTime(&time2, coloring_start, coloring_stop);
    cudaEventDestroy(coloring_start);
    cudaEventDestroy(coloring_stop);

    coloringTheMap1<<<1, SIZE>>>(heatmap);
    cudaDeviceSynchronize();

    ///////////////////////////////////////////////////

    cudaEvent_t scaling_start, scaling_stop;
    cudaEventCreate(&scaling_start);
    cudaEventCreate(&scaling_stop);
    cudaEventRecord(scaling_start, 0);

    scalingTheMap<<<1,SIZE>>>(heatmap, scaled_heatmap);

    cudaEventRecord(scaling_stop, 0);
    cudaEventSynchronize(scaling_stop);
    cudaEventElapsedTime(&time3, scaling_start, scaling_stop);
    cudaEventDestroy(scaling_start);
    cudaEventDestroy(scaling_stop);

    ///////////////////////////////////////////////////

    const int w[5][5] = {{1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};

    cudaEvent_t filtering_start, filtering_stop;
    cudaEventCreate(&filtering_start);
    cudaEventCreate(&filtering_stop);
    cudaEventRecord(filtering_start, 0);

    filteringTheMap<<<1,SIZE>>>(scaled_heatmap, blurred_heatmap, w);

    cudaEventRecord(filtering_stop, 0);
    cudaEventSynchronize(filtering_stop);
    cudaEventElapsedTime(&time4, filtering_start, filtering_stop);
    cudaEventDestroy(filtering_start);
    cudaEventDestroy(filtering_stop);
  }

}  // namespace Ped
