#include "ped_agent_soa.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace Ped {

void* MallocPinned(std::size_t bytes) {
  void* result;
  cudaMallocHost(&result, bytes);
  return result;
}

void FreePinned(void* mem) {
  cudaFreeHost(mem);
}

}  // namespace Ped