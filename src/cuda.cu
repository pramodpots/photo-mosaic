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
unsigned int cuda_TILES_X, cuda_TILES_Y, cuda_input_image_width, cuda_input_image_height, cuda_input_image_channels;

// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
unsigned long long* h_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
unsigned char* h_mosaic_value;
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

    cuda_input_image_width = input_image->width;
    cuda_input_image_height = input_image->height;
    cuda_input_image_channels = input_image->channels;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));
    CUDA_CALL(cudaMemset(d_mosaic_sum, 0, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

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
    CUDA_CALL(cudaMemset(d_global_pixel_sum, 0, input_image->channels * sizeof(unsigned long long)));
}

__global__ void tile_sum_CUDA(unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum, unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, unsigned int cuda_input_image_width, unsigned int cuda_input_image_height, unsigned int cuda_input_image_channels) {
    __shared__ long long int sm[3];

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int op_idx = x + y * blockDim.x * gridDim.x;
    unsigned long long pixel_idx = op_idx * cuda_input_image_channels;

    //unsigned int idx = ix + iy * nx;

    //int offset = t_x + t_y * blockDim.x * gridDim.x;

    const unsigned int tile_index = (blockIdx.y * cuda_TILES_X + blockIdx.x) * cuda_input_image_channels;

    atomicAdd(&d_mosaic_sum[tile_index + 0], d_input_image_data[pixel_idx + 0]);
    atomicAdd(&d_mosaic_sum[tile_index + 1], d_input_image_data[pixel_idx + 1]);
    atomicAdd(&d_mosaic_sum[tile_index + 2], d_input_image_data[pixel_idx + 2]);

    //__syncthreads();
    //printf("thread_id (%d,%d) block_id (%d,%d) thread_x_y (%d,%d) op_idx %2d  tile_index %2d\n", 
        //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, x, y, pixel_idx, tile_index);
}

void cuda_stage1() {
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 32 x 32
    //int warps_per_grid = cuda_input_image.width / TILE_SIZE;

    //print_device_arch();

    tile_sum_CUDA <<<blocksPerGrid, threadsPerBlock >>>(d_input_image_data, d_mosaic_sum, cuda_TILES_X, cuda_TILES_Y, cuda_input_image_width, cuda_input_image_height, cuda_input_image_channels);
    /* wait for all threads to complete */
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    h_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels * sizeof(unsigned long long));
    cudaMemcpy(h_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    validate_tile_sum(&cuda_input_image, h_mosaic_sum);
#endif
}

__global__ void compact_mosaic(unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum, unsigned long long* d_global_pixel_sum, unsigned int cuda_input_image_channels) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    d_mosaic_value[index] = (unsigned char)(d_mosaic_sum[index] / TILE_PIXELS);
    //d_mosaic_value[index + 1] = (unsigned char)(d_mosaic_sum[index + 1] / TILE_PIXELS);
    //d_mosaic_value[index + 2] = (unsigned char)(d_mosaic_sum[index + 2] / TILE_PIXELS);

    atomicAdd(&d_global_pixel_sum[index % cuda_input_image_channels], d_mosaic_value[index]);
    //atomicAdd(&d_global_pixel_sum[index + 1], d_mosaic_value[index + 1]);
    //atomicAdd(&d_global_pixel_sum[index + 2], d_mosaic_value[index + 2]);
    //d_global_pixel_sum[0] += d_mosaic_value[index + 0];
   // d_global_pixel_sum[1] += d_mosaic_value[index + 1];
   // d_global_pixel_sum[2] += d_mosaic_value[index + 2];
}


void cuda_stage2(unsigned char* output_global_average) {

    long long int total_pixels = cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels;

    
    dim3 blocksPerGrid(total_pixels / (TILE_SIZE * TILE_SIZE), 1, 1);
    dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE, 1, 1);  // 32 x 32 ==> 1024

    compact_mosaic << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_value, d_mosaic_sum, d_global_pixel_sum, cuda_input_image_channels);
    /* wait for all threads to complete */
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    unsigned long long* h_global_pixel_sum;
    unsigned int size = cuda_input_image_channels * sizeof(unsigned long long);
    /* allocate the host memory */
    h_global_pixel_sum = (unsigned long long*)malloc(size);
    cudaMemcpy(h_global_pixel_sum, d_global_pixel_sum, size, cudaMemcpyDeviceToHost);

    for (int ch = 0; ch < cuda_input_image_channels; ++ch) {
        output_global_average[ch] = (unsigned char)(h_global_pixel_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
        //printf("global_average= %d\n", h_global_pixel_sum[ch]);
    }
    
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    h_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels * sizeof(unsigned char));
    cudaMemcpy(h_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image_channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);
#endif    
}

__global__ void cuda_broadcast(unsigned char* d_output_image_data, unsigned char* d_mosaic_value, unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, unsigned int cuda_input_image_width, unsigned int cuda_input_image_height, unsigned int cuda_input_image_channels) {
    //int t_x = threadIdx.x + blockIdx.x * blockDim.x;
    //int t_y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = t_x + t_y * blockDim.x * gridDim.x;

    //const unsigned int tile_index = (t_y * cuda_TILES_X + t_x) * cuda_input_image_channels;
    //const unsigned int tile_offset = (t_y * cuda_TILES_Y * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cuda_input_image_channels;
    //const unsigned char pixel_2 = d_input_image_data[offset];
    //atomicAdd(&d_mosaic_sum[tile_index + 0], d_input_image_data[offset + 0]);
    //atomicAdd(&d_mosaic_sum[tile_index + 1], d_input_image_data[offset + 1]);
    //atomicAdd(&d_mosaic_sum[tile_index + 2], d_input_image_data[offset + 2]);

    int t_x = blockIdx.x;
    int t_y = blockIdx.y;
    int p_x = threadIdx.x;
    int p_y = threadIdx.y;

    //int offset = t_x + t_y * blockDim.x * gridDim.x;

    const unsigned int tile_index = (t_y * cuda_TILES_X + t_x) * cuda_input_image_channels;
    const unsigned int tile_offset = (t_y * cuda_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cuda_input_image_channels;

    const unsigned int pixel_offset = (p_y * cuda_input_image_width + p_x) * cuda_input_image_channels;

    d_output_image_data[tile_offset + pixel_offset + 0] = d_mosaic_value[tile_index + 0];
    d_output_image_data[tile_offset + pixel_offset + 1] = d_mosaic_value[tile_index + 1];
    d_output_image_data[tile_offset + pixel_offset + 2] = d_mosaic_value[tile_index + 2];
}
void cuda_stage3() {
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 32 x 32
    //dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE, 1, 1);  // 32 x 32
    //dim3 blocksPerGrid((cuda_TILES_X * cuda_TILES_Y) / (TILE_SIZE * TILE_SIZE), 1, 1);

    cuda_broadcast << <blocksPerGrid, threadsPerBlock >> > (d_output_image_data, d_mosaic_value, cuda_TILES_X, cuda_TILES_Y, cuda_input_image_width, cuda_input_image_height, cuda_input_image_channels);
    /* wait for all threads to complete */
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);

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
}
