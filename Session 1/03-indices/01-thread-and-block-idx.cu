#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  } 
  else {
    // printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   */

  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();
  // Don't forget that the kernel execution is asynchronous, and you must sync on its completion
  
  cudaDeviceSynchronize();
}
