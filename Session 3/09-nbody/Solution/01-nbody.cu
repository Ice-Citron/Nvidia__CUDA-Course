#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 256

typedef struct { float x, y, z, vx, vy, vz; } Body;

__global__ void bodyForce_kernel(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx;
        p[i].vy += dt*Fy;
        p[i].vz += dt*Fz;
    }
}

__global__ void integrate_positions_kernel(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx*dt;
        p[i].y += p[i].vy*dt;
        p[i].z += p[i].vz*dt;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2<<11;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);

    const char *initialized_values;
    const char *solution_values;

    if (nBodies == 2<<11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else { // nBodies == 2<<15
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf = (float*)malloc(bytes);
    Body *p = (Body*)buf;

    read_values_from_file(initialized_values, buf, bytes);

    // Allocate memory on the GPU
    Body *d_bodies;
    cudaMalloc(&d_bodies, bytes);
    cudaMemcpy(d_bodies, p, bytes, cudaMemcpyHostToDevice);

    double totalTime = 0.0;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        bodyForce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, dt, nBodies);
        cudaDeviceSynchronize();

        integrate_positions_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, dt, nBodies);
        cudaDeviceSynchronize();

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    cudaMemcpy(p, d_bodies, bytes, cudaMemcpyDeviceToHost);

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, buf, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    free(buf);
    cudaFree(d_bodies);

    return 0;
}