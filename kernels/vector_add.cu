#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = a[index] + b[index];
}

int main() {
  constexpr int n = 1024;
  std::vector<float> a(n, 1.0F), b(n, 2.0F), out(n, 0.0F);
  float *device_a = nullptr, *device_b = nullptr, *device_out = nullptr;
  cudaMalloc(&device_a, n * sizeof(float));
  cudaMalloc(&device_b, n * sizeof(float));
  cudaMalloc(&device_out, n * sizeof(float));
  cudaMemcpy(device_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  vector_add<<<(n + 255) / 256, 256>>>(device_a, device_b, device_out, n);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(), device_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  std::printf("out[0]=%.1f out[%d]=%.1f\n", out[0], n - 1, out[n - 1]);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_out);
  return (out[0] == 3.0F && out[n - 1] == 3.0F) ? 0 : 1;
}
