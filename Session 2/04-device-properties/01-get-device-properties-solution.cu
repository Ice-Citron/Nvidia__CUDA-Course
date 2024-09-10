#include <stdio.h>

int main()
{
  /*
   * Device ID is required first to query the device.
   */

  int deviceId;
  cudaGetDevice(&deviceId); // gets CUDA device ID

  cudaDeviceProp props; // cudaDeviceProp <-- custom data type for storing properties of CUDA device
  cudaGetDeviceProperties(&props, deviceId); // this method fetches the properties of device::0 (specified by deviceID) and stores them in the prop Sturtcure

  /*
   * `props` now contains several properties about the current device.
   */

  int computeCapabilityMajor = props.major; // indicate version of CUDA compatibility supported by device
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount = props.multiProcessorCount; // SM=streaming multiprocessors // give counts of SMs og GPU
  int warpSize = props.warpSize; // indicate number of threads that execute concurrently in a warp.


  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
