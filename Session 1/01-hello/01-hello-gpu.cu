#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * Refactor the `helloGPU` definition to be a kernel
 * that can be launched on the GPU. Update its message
 * to read "Hello from the GPU!"
 */

__global__ void helloGPU()
{
  printf("Hello from the gPU.\n");
}

int main()
{

  helloCPU();

  /*
   * Refactor this call to `helloGPU` so that it launches
   * as a kernel on the GPU.
   */

  helloGPU<<<1, 1>>>(); // launches kernel with 1 block and 1 thread

  /*
   * Add code below to synchronize on the completion of the
   * `helloGPU` kernel completion before continuing the CPU
   * thread.
   */
   
  cudaDeviceSynchronize(); // waits for GPU to finish before accessing on host
}

// nvcc -o helloGPU 01-hello-gpu.cu      // <-- compiles the .cu file into executable named "helloGPU" <-- NVCC means Nvidia Cuda Compiler
// ./helloGPU     // <-- to execute the executable named "helloGPU"