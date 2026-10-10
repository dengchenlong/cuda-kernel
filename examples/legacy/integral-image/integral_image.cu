#include <iostream>
#include <cstdlib> // For rand()
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

// 基础实现
__global__ void computeIntegralImageKernel(const unsigned char* inputImage, unsigned int* integralImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int sum = 0;
        for (int i = 0; i <= y; ++i) {
            for (int j = 0; j <= x; ++j) {
                sum += inputImage[i * width + j];
            }
        }
        integralImage[y * width + x] = sum;
    }
}

// 基础实现
__global__ void prefixSumKernel1(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height)
    {
        return;
    }

    const uint8_t* indexRowImg = img + y * imgStep;
    uint* indexRowIntegral = integral + y * integralStep;
    uint accRow = 0;
    for (int i = 0; i < width; ++i)
    {
        accRow += indexRowImg[i];
        indexRowIntegral[i] = accRow;
    }
}

// 向量化访存+循环展开
__global__ void prefixSumKernel2(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height)
    {
        return;
    }

    const uint8_t* indexRowImg = img + y * imgStep;
    uint* indexRowIntegral = integral + y * integralStep;
    uint accRow = 0;
    int i = 0;
    for (; i < width; i += 4)
    {
        uchar4 v = *(uchar4*)(indexRowImg + i);
        accRow += v.x;
        *indexRowIntegral++ = accRow;
        accRow += v.y;
        *indexRowIntegral++ = accRow;
        accRow += v.z;
        *indexRowIntegral++ = accRow;
        accRow += v.w;
        *indexRowIntegral++ = accRow;
    }
    for (; i < width; ++i)
    {
        accRow += indexRowImg[i];
        *indexRowIntegral++ = accRow;
    }
}

// 向量化访存+循环展开
__global__ void prefixSumKernel(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height)
    {
        return;
    }

    const uint8_t* indexRowImg = img + y * imgStep;
    uint* indexRowIntegral = integral + y * integralStep;
    uint accRow = 0;
    int i = 0;
    for (; i < width; i += 16)
    {
        uint4 v = *(uint4*)(indexRowImg + i);

#define CODE(n) \
        accRow += reinterpret_cast<uint8_t*>(&v)[n];\
        *indexRowIntegral++ = accRow

        CODE(0);
        CODE(1);
        CODE(2);
        CODE(3);
        CODE(4);
        CODE(5);
        CODE(6);
        CODE(7);
        CODE(8);
        CODE(9);
        CODE(10);
        CODE(11);
        CODE(12);
        CODE(13);
        CODE(14);
        CODE(15);

#undef CODE

    }
    for (; i < width; ++i)
    {
        accRow += indexRowImg[i];
        *indexRowIntegral++ = accRow;
    }
}

// 啥都不干
__global__ void prefixSumKernel4(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
}

// 基础实现+指针优化
__global__ void prefixSumKernel5(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height)
    {
        return;
    }

    const uint8_t* indexRowImgPtr = img + y * imgStep;
    uint* indexRowIntegralPtr = integral + y * integralStep;
    const uint8_t* indexRowImgEnd = img + y * imgStep + width;
    uint accRow = 0;
    for (; indexRowImgPtr < indexRowImgEnd; ++indexRowImgPtr, ++indexRowIntegralPtr)
    {
        accRow += *indexRowImgPtr;
        *indexRowIntegralPtr = accRow;
    }
}

// 归约
__global__ void prefixSumKernel6(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    integral[y * integralStep + x] = img[y * imgStep + x];
    __syncthreads();

    for (int stride = 1; stride < width; stride <<= 1)
    {
        if (x >= stride)
        {
            const auto a = integral[y * integralStep + x - stride];
            const auto b = integral[y * integralStep + x];
            __syncthreads();
            integral[y * integralStep + x] = a + b;
            __syncthreads();
        }
    }
}

// 共享内存+归约
__global__ void prefixSumKernel7(const uint8_t* img, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    __shared__ uint32_t s_data[1024];
    s_data[threadIdx.x] = img[y * imgStep + x];
    __syncthreads();

    for (int stride = 1; stride < width; stride <<= 1)
    {
        if (x >= stride)
        {
            const auto a = s_data[x - stride];
            const auto b = s_data[x];
            __syncthreads();
            s_data[x] = a + b;
            __syncthreads();
        }
    }

    integral[y * integralStep + x] = s_data[threadIdx.x];
}

// 基础实现
__global__ void columnSumKernel1(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    uint accCol = 0;
    int i = x;
    for (int i = x; i < imgStep * height; i += imgStep)
    {
        accCol += prefix[i];
        integral[i] = accCol;
    }
}

// 基础实现+指针优化
__global__ void columnSumKernel(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    uint accCol = 0;
    auto prefixPtr = prefix + x;
    auto integralPtr = integral + x;
    auto prefixEnd = prefix + imgStep * height;
    for (; prefixPtr < prefixEnd; prefixPtr += imgStep, integralPtr += integralStep)
    {
        accCol += *prefixPtr;
        *integralPtr = accCol;
    }
}

// 只有循环
__global__ void columnSumKernel3(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    for (int i = x; i < imgStep * height; i += imgStep)
    {
        prefix[i];
    }
}

// 只有访问
__global__ void columnSumKernel4(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    prefix[x];
}

// 啥都不干
__global__ void columnSumKernel5(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // if (x >= width)
    // {
    //     return;
    // }

    // for (int i = x; i < imgStep * height; i += imgStep)
    // {
    // }
}

// 只有循环+循环展开
__global__ void columnSumKernel6(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    for (int i = x; i < imgStep * height; i += imgStep * 8)
    {
        prefix[i];
        prefix[i + imgStep];
        prefix[i + imgStep * 2];
        prefix[i + imgStep * 3];
        prefix[i + imgStep * 4];
        prefix[i + imgStep * 5];
        prefix[i + imgStep * 6];
        prefix[i + imgStep * 7];
    }
}

// 基础实现+循环展开
__global__ void columnSumKernel7(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    uint accCol = 0;
    for (int i = x; i < imgStep * height; i += imgStep * 8)
    {
        accCol += prefix[i];
        integral[i] = accCol;
        accCol += prefix[i + imgStep];
        integral[i + 1] = accCol;
        accCol += prefix[i + imgStep * 2];
        integral[i + 2] = accCol;
        accCol += prefix[i + imgStep * 3];
        integral[i + 3] = accCol;
        accCol += prefix[i + imgStep * 4];
        integral[i + 4] = accCol;
        accCol += prefix[i + imgStep * 5];
        integral[i + 5] = accCol;
        accCol += prefix[i + imgStep * 6];
        integral[i + 6] = accCol;
        accCol += prefix[i + imgStep * 7];
        integral[i + 7] = accCol;
    }
}

// 基础实现+循环展开+删除数据依赖
__global__ void columnSumKernel8(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
    {
        return;
    }

    for (int i = x; i < imgStep * height; i += imgStep * 8)
    {
        integral[i] = prefix[i];
        integral[i + 1] = prefix[i + 1];
        integral[i + 2] = prefix[i + 2];
        integral[i + 3] = prefix[i + 3];
        integral[i + 4] = prefix[i + 4];
        integral[i + 5] = prefix[i + 5];
        integral[i + 6] = prefix[i + 6];
        integral[i + 7] = prefix[i + 7];
    }
}

// 基础实现+向量化访存+循环展开
__global__ void columnSumKernel9(const uint32_t* prefix, size_t imgStep, uint32_t* integral, size_t integralStep, const uint32_t width, const uint32_t height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= imgStep)
    {
        return;
    }

    uint accCol1 = 0;
    uint accCol2 = 0;
    uint accCol3 = 0;
    uint accCol4 = 0;
    for (int row = 0; row < height; ++row)
    {
        uint4 v = *(uint4*)(prefix + imgStep * row + idx * 4);
        accCol1 += v.x;
        integral[integralStep * row + idx * 4] = accCol1;
        accCol2 += v.y;
        integral[integralStep * row + idx * 4 + 1] = accCol2;
        accCol3 += v.z;
        integral[integralStep * row + idx * 4 + 2] = accCol3;
        accCol4 += v.w;
        integral[integralStep * row + idx * 4 + 3] = accCol4;
    }
}

std::vector<uint> computeIntegralImageGold(const uint8_t* img, const uint width, const uint height) {
    return columnSumGold(prefixSumGold(img, width, height).data(), width, height);
}

std::vector<uint> prefixSumGold(const uint8_t* img, const uint width, const uint height) {
    std::vector<uint> integral(width * height);
    for (uint y = 0; y < height; ++y) {
        uint accRow = 0;
        for (uint x = 0; x < width; ++x) {
            accRow += img[y * width + x];
            integral[y * width + x] = accRow;
        }
    }
    return integral;
}

std::vector<uint> columnSumGold(const uint* prefix, const uint width, const uint height) {
    std::vector<uint> integral(width * height);
    for (uint x = 0; x < width; ++x) {
        uint accCol = 0;
        for (uint y = 0; y < height; ++y) {
            accCol += prefix[y * width + x];
            integral[y * width + x] = accCol;
        }
    }
    return integral;
}

void computeIntegralImageCUDA(const unsigned char* d_inputImage, unsigned int* d_integralImage, int width, int height) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    computeIntegralImageKernel<<<gridSize, blockSize>>>(d_inputImage, d_integralImage, width, height);
}

void prefixSumCUDA1(const unsigned char* d_inputImage, size_t inputStep, uint* d_integralImage, size_t integralStep, int width, int height) {
    dim3 gridSize(1, 64);
    dim3 blockSize(1, (height + gridSize.y - 1) / gridSize.y);
    prefixSumKernel<<<gridSize, blockSize>>>(d_inputImage, inputStep, d_integralImage, integralStep, width, height);
}

void prefixSumCUDA(const unsigned char* d_inputImage, size_t inputStep, uint* d_integralImage, size_t integralStep, int width, int height) {
    dim3 gridSize(1, height);
    dim3 blockSize(width, 1);
    prefixSumKernel<<<gridSize, blockSize>>>(d_inputImage, inputStep, d_integralImage, integralStep, width, height);
}

void columnSumCUDA(const uint* d_prefix, size_t inputStep, uint* d_integralImage, size_t integralStep, int width, int height) {
    dim3 gridSize(64, 1);
    dim3 blockSize((width + gridSize.x - 1) / gridSize.x, 1);
    columnSumKernel<<<gridSize, blockSize>>>(d_prefix, inputStep, d_integralImage, integralStep, width, height);
}

void columnSumCUDA2(const uint* d_prefix, size_t inputStep, uint* d_integralImage, size_t integralStep, int width, int height) {
    dim3 gridSize(1, 1);
    dim3 blockSize(200, 1);
    columnSumKernel<<<gridSize, blockSize>>>(d_prefix, inputStep, d_integralImage, integralStep, width, height);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void loadImage(const char* filename, unsigned char** image, int* width, int* height) {
    // Example implementation: Set dummy values
    *width = 800; // Replace with actual image width
    *height = 250; // Replace with actual image height
    size_t imageSize = (*width) * (*height);
    *image = (unsigned char*)malloc(imageSize);
    if (*image == nullptr) {
        std::cerr << "Failed to allocate memory for image" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Fill the image with dummy data (replace with actual image loading logic)
    memset(*image, 255, imageSize);
}

template <typename T>
void generateRandomImage(T** image, int width, int height) {
    // Set image dimensions
    size_t imageSize = (width) * (height);

    // Allocate memory for the image
    *image = (T*)malloc(imageSize * sizeof(T));
    if (*image == nullptr) {
        std::cerr << "Failed to allocate memory for image" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Fill the image with random data
    for (size_t i = 0; i < imageSize; ++i) {
        (*image)[i] = rand() % 256; // Random value between 0 and 255
    }
}

int testIntegralImage(const int width, const int height) {
    unsigned char* h_image; // Host image data

    generateRandomImage(&h_image, width, height);

    // Allocate device memory
    unsigned char* d_image;
    unsigned int* d_integralImage;
    size_t imageSize = width * height * sizeof(unsigned char);
    size_t integralImageSize = width * height * sizeof(int);

    size_t imageStep;
    checkCudaError(cudaMallocPitch((void**)&d_image, imageSize), "Allocating device image");
    size_t integralStep;
    checkCudaError(cudaMallocPitch((void**)&d_integralImage, integralImageSize), "Allocating device integral image");

    // Copy image data to device
    checkCudaError(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice), "Copying image to device");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");

    // Record the start event
    checkCudaError(cudaEventRecord(start, 0), "Recording start event");

    // Launch the integral image kernel
    computeIntegralImageCUDA(d_image, d_integralImage, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Record the stop event
    checkCudaError(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

    // Calculate elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Destroy CUDA events
    checkCudaError(cudaEventDestroy(start), "Destroying start event");
    checkCudaError(cudaEventDestroy(stop), "Destroying stop event");

    // Copy the integral image back to host
    uint* h_integralImage = new uint[width * height];
    checkCudaError(cudaMemcpy(h_integralImage, d_integralImage, integralImageSize, cudaMemcpyDeviceToHost), "Copying integral image to host");

    // Clean up
    delete[] h_integralImage;
    cudaFree(d_image);
    cudaFree(d_integralImage);
    free(h_image);

    return EXIT_SUCCESS;
}

int testPrefixSum(const int width, const int height) {
    unsigned char* h_image; // Host image data

    generateRandomImage(&h_image, width, height);

    // Allocate device memory
    unsigned char* d_image;
    unsigned int* d_integralImage;

    size_t imageStep;
    checkCudaError(cudaMallocPitch((void**)&d_image, &imageStep, width, height), "Allocating device image");
    size_t integralStep;
    checkCudaError(cudaMallocPitch((void**)&d_integralImage, &integralStep, width * sizeof(uint), height), "Allocating device integral image");
    integralStep /= sizeof(uint);

    // Copy image data to device
    checkCudaError(cudaMemcpy2D(d_image, imageStep, h_image, width, width, height, cudaMemcpyHostToDevice), "Copying image to device");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");

    // Record the start event
    checkCudaError(cudaEventRecord(start, 0), "Recording start event");

    // Launch the integral image kernel
    prefixSumCUDA(d_image, imageStep, d_integralImage, integralStep, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Record the stop event
    checkCudaError(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

    // Calculate elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Destroy CUDA events
    checkCudaError(cudaEventDestroy(start), "Destroying start event");
    checkCudaError(cudaEventDestroy(stop), "Destroying stop event");

    // Copy the integral image back to host
    int* h_integralImage = new int[width * height];
    checkCudaError(cudaMemcpy2D(h_integralImage, sizeof(int) * width, d_integralImage, sizeof(int) * integralStep, sizeof(int) * width, height, cudaMemcpyDeviceToHost), "Copying integral image to host");

    auto gold = prefixSumGold(h_image, width, height);
    memcmp(gold.data(), h_integralImage, sizeof(int) * width * height) == 0 ? std::cout << "Test passed!\n" : std::cout << "Test failed!\n";

    // Clean up
    delete[] h_integralImage;
    cudaFree(d_image);
    cudaFree(d_integralImage);
    free(h_image);

    return EXIT_SUCCESS;
}

int testColumnSum(const int width, const int height) {
    unsigned int* h_image; // Host image data

    generateRandomImage(&h_image, width, height);

    // Allocate device memory
    unsigned int* d_image;
    unsigned int* d_integralImage;

    size_t imageStep;
    checkCudaError(cudaMallocPitch((void**)&d_image, &imageStep, width * sizeof(uint), height), "Allocating device image");
    imageStep /= sizeof(uint);
    size_t integralStep;
    checkCudaError(cudaMallocPitch((void**)&d_integralImage, &integralStep, width * sizeof(uint), height), "Allocating device integral image");
    integralStep /= sizeof(uint);

    // Copy image data to device
    checkCudaError(cudaMemcpy2D(d_image, imageStep * sizeof(uint), h_image, width * sizeof(uint), width * sizeof(uint), height, cudaMemcpyHostToDevice), "Copying image to device");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");

    // Record the start event
    checkCudaError(cudaEventRecord(start, 0), "Recording start event");

    // Launch the integral image kernel
    columnSumCUDA(d_image, imageStep, d_integralImage, integralStep, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Record the stop event
    checkCudaError(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

    // Calculate elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Destroy CUDA events
    checkCudaError(cudaEventDestroy(start), "Destroying start event");
    checkCudaError(cudaEventDestroy(stop), "Destroying stop event");

    // Copy the integral image back to host
    int* h_integralImage = new int[width * height];
    checkCudaError(cudaMemcpy2D(h_integralImage, sizeof(int) * width, d_integralImage, sizeof(int) * integralStep, sizeof(int) * width, height, cudaMemcpyDeviceToHost), "Copying integral image to host");

    auto gold = columnSumGold(h_image, width, height);
    memcmp(gold.data(), h_integralImage, sizeof(int) * width * height) == 0 ? std::cout << "Test passed!\n" : std::cout << "Test failed!\n";

    // Clean up
    delete[] h_integralImage;
    cudaFree(d_image);
    cudaFree(d_integralImage);
    free(h_image);

    return EXIT_SUCCESS;
}

int main(){
    return testColumnSum(800, 2500);
}