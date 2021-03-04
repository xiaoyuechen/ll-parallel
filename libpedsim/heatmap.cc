#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ped_model.h"

namespace Ped {

__global__ void InitHm(int* hm) {
  // int tid =
}

void Model::setupHeatmapCuda() {
  // cudaStream_t s[6];
  // for(int i = 0; i != 6; ++i) {
  //     cudaStreamCreate(s + i);
  // }

  int *hm, *shm, *bhm;

  cudaMallocAsync(&hm, SIZE * SIZE * sizeof(int));
  cudaMallocAsync(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  cudaMallocAsync(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  cudaMallocAsync(&heatmap, SIZE * sizeof(int*));
  cudaMallocAsync(&scaled_heatmap, SCALED_SIZE * sizeof(int*));
  cudaMallocAsync(&desired_xs, agents->size() * sizeof(int));
  cudaMallocAsync(&desired_ys, agents->size() * sizeof(int));

  cudaMallocHost(&blurred_heatmap, SCALED_SIZE * sizeof(int*));

  // cudaMalloc()
}

}  // namespace Ped
