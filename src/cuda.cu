#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda.cuh"

#include <cstring>

#include "helper.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// unsigned long long* h_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// unsigned char* h_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));
}

__global__ void tile_sum_CUDA_shuffle(unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum) {
    // Sum pixel data within each tile

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_idx = x + y * blockDim.x * gridDim.x;
    // idx for pixel in actual image
    unsigned int gbl_pixel_idx = global_idx * CHANNELS;
    // idx of current tile
    unsigned int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * CHANNELS;

    // load pixel values into local variables
    unsigned long long r = d_input_image_data[gbl_pixel_idx + 0];
    unsigned long long g = d_input_image_data[gbl_pixel_idx + 1];
    unsigned long long b = d_input_image_data[gbl_pixel_idx + 2];
   
    // shuffle down
    for (int offset = 16; offset > 0; offset >>= 1) {
        r += __shfl_down(r, offset);
        g += __shfl_down(g, offset);
        b += __shfl_down(b, offset);
    }

    // write to global result using atomics if first thread in warp
    // block size is 32 x 32 so each wrap starts at 0
    if (threadIdx.x == 0) {
        atomicAdd(&d_mosaic_sum[tile_index + 0], r);
        atomicAdd(&d_mosaic_sum[tile_index + 1], g);
        atomicAdd(&d_mosaic_sum[tile_index + 2], b);
    }
}

void cuda_stage1() {
    // create 2d block of size equal to tiles
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);
    // threads per block 32 x 32 = 1024
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // Kernel call
    tile_sum_CUDA_shuffle << <blocksPerGrid, threadsPerBlock>> > (d_input_image_data, d_mosaic_sum);
    
    // sync 
    cudaDeviceSynchronize();

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // h_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels * sizeof(unsigned long long));
    // cudaMemcpy(h_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    // validate_tile_sum(&cuda_input_image, h_mosaic_sum);
#endif
}

__global__ void compact_mosaic_shuffle(unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum, unsigned long long* d_global_pixel_sum) {
    // Calculate the average of each tile, and sum these to produce a whole image average.

    int idx = threadIdx.x;
    int offset_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // calculate and load values into d_mosaic_value
    d_mosaic_value[offset_idx * CHANNELS + 0] = (unsigned char)(d_mosaic_sum[offset_idx * CHANNELS + 0] / TILE_PIXELS);
    d_mosaic_value[offset_idx * CHANNELS + 1] = (unsigned char)(d_mosaic_sum[offset_idx * CHANNELS + 1] / TILE_PIXELS);
    d_mosaic_value[offset_idx * CHANNELS + 2] = (unsigned char)(d_mosaic_sum[offset_idx * CHANNELS + 2] / TILE_PIXELS);

    // wait until each value is loaded
    __syncthreads();

    // load r,g,b locally per thread
    unsigned long long r = d_mosaic_value[offset_idx * CHANNELS + 0];
    unsigned long long g = d_mosaic_value[offset_idx * CHANNELS + 1];
    unsigned long long b = d_mosaic_value[offset_idx * CHANNELS + 2];

    // shuffle down
    for (int offset = 16; offset > 0; offset >>= 1) {
        r += __shfl_down(r, offset);
        g += __shfl_down(g, offset);
        b += __shfl_down(b, offset);
    }

    // write to global result using atomics if first thread in warp
    // block size is 32 so each wrap starts at 0
    if (threadIdx.x == 0) {
        atomicAdd(&d_global_pixel_sum[0], r);
        atomicAdd(&d_global_pixel_sum[1], g);
        atomicAdd(&d_global_pixel_sum[2], b);
    }
}

void cuda_stage2(unsigned char* output_global_average) {
    int compact_mosaic_pixels = cuda_TILES_X * cuda_TILES_Y * CHANNELS;
    // create 1D blocks
    dim3 blocksPerGrid((cuda_TILES_X * cuda_TILES_Y) / TILE_SIZE, 1, 1);
    dim3 threadsPerBlock(TILE_SIZE, 1, 1);
    // kernel call
    compact_mosaic_shuffle <<<blocksPerGrid, threadsPerBlock>>> (d_mosaic_value, d_mosaic_sum, d_global_pixel_sum);

    // local host variable for calculating global avg
    unsigned long long* h_global_pixel_sum;
    // allocate pinned memory
    cudaMallocHost((unsigned long long**)&h_global_pixel_sum, sizeof(unsigned long long) * CHANNELS);
    // copy calculated global_pixel_sum to host 
    cudaMemcpy(h_global_pixel_sum, d_global_pixel_sum, sizeof(unsigned long long) * CHANNELS, cudaMemcpyDeviceToHost);
    
    // calculate and save into output_global_average
    // compiler is better at optimizing this loop as const CHANNELS used.
    // tried loop unrolling here but it degrades performance
    for (int ch = 0; ch < CHANNELS; ++ch) {
        output_global_average[ch] = (unsigned char)(h_global_pixel_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
    }
    // free pinned memory
    cudaFreeHost(h_global_pixel_sum);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // h_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * CHANNELS * sizeof(unsigned char));
    // cudaMemcpy(h_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);
#endif    
}

__global__ void cuda_broadcast(unsigned char* d_output_image_data, unsigned char* d_mosaic_value) {
    // Broadcast the compact mosaic pixels back out to the full image size

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int global_idx = x + y * blockDim.x * gridDim.x;
    int gbl_pixel_idx = global_idx * CHANNELS;
    int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * CHANNELS;

    // tile_index will be same for block. 
    // broadcast same values into all block indexes
    d_output_image_data[gbl_pixel_idx + 0] = d_mosaic_value[tile_index + 0];
    d_output_image_data[gbl_pixel_idx + 1] = d_mosaic_value[tile_index + 1];
    d_output_image_data[gbl_pixel_idx + 2] = d_mosaic_value[tile_index + 2];
}

void cuda_stage3() {
    // create blocks equal to tiles
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // kernel call
    cuda_broadcast << <blocksPerGrid, threadsPerBlock >> > (d_output_image_data, d_mosaic_value);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_broadcast(&input_image, mosaic_value, &output_image);
#endif    
}

void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_global_pixel_sum));
}
