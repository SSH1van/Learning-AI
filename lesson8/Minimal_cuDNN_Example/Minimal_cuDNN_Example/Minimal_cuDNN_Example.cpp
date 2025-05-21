#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0);
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;

    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = n * c * h * w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    float* x;
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float));
    for (int i = 0; i < NUM_ELEMENTS; i++) x[i] = i * 1.0f;
    std::cout << "Original array: ";
    for (int i = 0; i < NUM_ELEMENTS; i++) std::cout << x[i] << " ";
    std::cout << std::endl;

    float alpha[1] = { 1.0f };
    float beta[1] = { 0.0f };
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0f);

    cudnnActivationForward(handle_, sigmoid_activation, alpha, x_desc, x, beta, x_desc, x);
    cudaDeviceSynchronize();

    std::cout << "New array: ";
    for (int i = 0; i < NUM_ELEMENTS; i++) std::cout << x[i] << " ";
    std::cout << std::endl;

    cudnnDestroyActivationDescriptor(sigmoid_activation);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroy(handle_);
    std::cout << "Destroyed cuDNN handle." << std::endl;
    cudaFree(x);

    return 0;
}