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
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned char* d_global_pixel_avg;

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

    // Allocate buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_avg, input_image->channels * sizeof(unsigned char)));
}

__global__ void tile_sum_CUDA(unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum, unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, unsigned int cuda_input_image_width, unsigned int cuda_input_image_height, unsigned int cuda_input_image_channels) {
    extern __shared__ unsigned long long sm[];


    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int global_idx = x + y * blockDim.x * gridDim.x;
    unsigned long long gbl_pixel_idx = global_idx * CHANNELS;

    unsigned int local_idx = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned long long lo_pixel_idx = local_idx * CHANNELS;

    sm[lo_pixel_idx + 0] = d_input_image_data[gbl_pixel_idx + 0];
    sm[lo_pixel_idx + 1] = d_input_image_data[gbl_pixel_idx + 1];
    sm[lo_pixel_idx + 2] = d_input_image_data[gbl_pixel_idx + 2];
    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sm[lo_pixel_idx + 0] += sm[lo_pixel_idx + (stride * CHANNELS) + 0];
            sm[lo_pixel_idx + 1] += sm[lo_pixel_idx + (stride * CHANNELS) + 1];
            sm[lo_pixel_idx + 2] += sm[lo_pixel_idx + (stride * CHANNELS) + 2];
        }
        __syncthreads();
    }
    //long long int r, g, b;

    //r = d_input_image_data[pixel_idx + 0];
    //g = d_input_image_data[pixel_idx + 1];
    //b = d_input_image_data[pixel_idx + 2];
    
    
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    // _shfl_down() has implicit warp synchronisation, so __syncthreads() is not required!
    //    r += __shfl_down(r, offset + 0);
    //    g += __shfl_down(g, offset + 1);
    //    b += __shfl_down(b, offset + 2);
    //}

    //__syncthreads();

    unsigned int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * CHANNELS;

    if (threadIdx.x == 0) {
        atomicAdd(&d_mosaic_sum[tile_index + 0], sm[lo_pixel_idx + 0]);
        atomicAdd(&d_mosaic_sum[tile_index + 1], sm[lo_pixel_idx + 1]);
        atomicAdd(&d_mosaic_sum[tile_index + 2], sm[lo_pixel_idx + 2]);
    }

    //atomicAdd(&d_mosaic_sum[tile_index + 0], d_input_image_data[pixel_idx + 0]);
    //atomicAdd(&d_mosaic_sum[tile_index + 1], d_input_image_data[pixel_idx + 1]);
    //atomicAdd(&d_mosaic_sum[tile_index + 2], d_input_image_data[pixel_idx + 2]);
    //__syncthreads();
    //d_mosaic_sum[tile_index + 0] += d_input_image_data[pixel_idx + 0];
    //d_mosaic_sum[tile_index + 0] += d_input_image_data[pixel_idx + 0];
    //d_mosaic_sum[tile_index + 0] += d_input_image_data[pixel_idx + 0];

    //__syncthreads();
    //if (blockIdx.x == 7 && blockIdx.y == 7) { //(threadIdx.x == 31 && threadIdx.y == 31) {
        //printf("thread_id (%d,%d) block_id (%d,%d) blockDim (%d,%d) gridDim (%d,%d) thread_x_y (%d,%d) global_idx %2d  tile_index %2d local_idx %d\n",
            //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y, x, y, pixel_idx, local_idx, tile_index);

        //printf("thread_id (%d,%d) block_id (%d,%d) blockDim (%d,%d) gridDim (%d,%d) tile_index %d local_idx %d lo_pixel_idx %llu gbl_pixel_idx %llu, global_idx %d tile_index %d\n",
            //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y, tile_index, local_idx, lo_pixel_idx, gbl_pixel_idx, global_idx, tile_index);
    //}
}

void cuda_stage1() {
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 32 x 32
    //int warps_per_grid = cuda_input_image.width / TILE_SIZE;

    //print_device_arch();

    tile_sum_CUDA <<<blocksPerGrid, threadsPerBlock, sizeof(unsigned long long) * TILE_SIZE * TILE_SIZE * CHANNELS >>>(d_input_image_data, d_mosaic_sum, cuda_TILES_X, cuda_TILES_Y, cuda_input_image_width, cuda_input_image_height, cuda_input_image_channels);
    
    //tile_sum_CUDA << <blocksPerGrid, threadsPerBlock>> > (d_input_image_data, d_mosaic_sum, cuda_TILES_X, cuda_TILES_Y, cuda_input_image_width, cuda_input_image_height, cuda_input_image_channels);

    /* wait for all threads to complete */
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    //printf("cuda_TILES_X %d, cuda_TILES_Y %d, x * y * ch = %llu\n", cuda_TILES_X, cuda_TILES_Y, cuda_TILES_X * cuda_TILES_Y * CHANNELS);
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

__global__ void compact_mosaic(unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum, unsigned long long* d_global_pixel_sum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned long long sm[3];
    sm[0] = 0;
    sm[1] = 0;
    sm[2] = 0;
    d_mosaic_value[idx] = (unsigned char)(d_mosaic_sum[idx] / TILE_PIXELS);
    __syncthreads();

    sm[idx % 3] += d_mosaic_value[idx];

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&d_global_pixel_sum[0], sm[0]);
        atomicAdd(&d_global_pixel_sum[1], sm[1]);
        atomicAdd(&d_global_pixel_sum[2], sm[2]);
    }

    //for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //    if (threadIdx.x < stride) {
    //        sm[idx * CHANNELS + 0] += sm[(idx + stride) * CHANNELS + 0];
    //        sm[idx * CHANNELS + 1] += sm[(idx + stride) * CHANNELS + 1];
    //        sm[idx * CHANNELS + 2] += sm[(idx + stride) * CHANNELS + 2];
    //    }
    //    __syncthreads();
    //}



    //sm[idx % 3] += d_mosaic_value[idx];

    //__syncthreads();
    //if (threadIdx.x == 0) {
    //    atomicAdd(&d_global_pixel_sum[idx % CHANNELS], sm[idx % CHANNELS]);
    //}
    //d_mosaic_value[index + 1] = (unsigned char)(d_mosaic_sum[index + 1] / TILE_PIXELS);
    //d_mosaic_value[index + 2] = (unsigned char)(d_mosaic_sum[index + 2] / TILE_PIXELS);

    //atomicAdd(&d_global_pixel_sum[idx % CHANNELS], d_mosaic_value[idx]);
    
    
    //atomicAdd(&d_global_pixel_sum[index + 1], d_mosaic_value[index + 1]);
    //atomicAdd(&d_global_pixel_sum[index + 2], d_mosaic_value[index + 2]);
    //d_global_pixel_sum[0] += d_mosaic_value[index + 0];
   // d_global_pixel_sum[1] += d_mosaic_value[index + 1];
   // d_global_pixel_sum[2] += d_mosaic_value[index + 2];

    //printf("idx %d\n", idx);
}

__global__ void calculate_global_sum(unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum, unsigned long long* d_global_pixel_sum, unsigned char* d_global_pixel_avg, int blocks) {
    extern __shared__ unsigned long long sm[];
    //extern __shared__ unsigned long long sm2[];
    int idx = threadIdx.x;
    int offset_idx = threadIdx.x + blockIdx.x * blockDim.x;

    //printf("threadIdx.x %2d blockIdx.x %2d blockDim.x%2d idx %2d\n", threadIdx.x, blockIdx.x, blockDim.x, idx);
    d_mosaic_value[offset_idx * CHANNELS + 0] = (unsigned char)(d_mosaic_sum[offset_idx * CHANNELS + 0] / TILE_PIXELS);
    d_mosaic_value[offset_idx * CHANNELS + 1] = (unsigned char)(d_mosaic_sum[offset_idx * CHANNELS + 1] / TILE_PIXELS);
    d_mosaic_value[offset_idx * CHANNELS + 2] = (unsigned char)(d_mosaic_sum[offset_idx * CHANNELS + 2] / TILE_PIXELS);
    sm[idx * CHANNELS + 0] = d_mosaic_value[offset_idx * CHANNELS + 0];
    sm[idx * CHANNELS + 1] = d_mosaic_value[offset_idx * CHANNELS + 1];
    sm[idx * CHANNELS + 2] = d_mosaic_value[offset_idx * CHANNELS + 2];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sm[idx * CHANNELS + 0] += sm[(idx + stride) * CHANNELS + 0];
            sm[idx * CHANNELS + 1] += sm[(idx + stride) * CHANNELS + 1];
            sm[idx * CHANNELS + 2] += sm[(idx + stride) * CHANNELS + 2];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&d_global_pixel_sum[0], sm[0]);
        atomicAdd(&d_global_pixel_sum[1], sm[1]);
        atomicAdd(&d_global_pixel_sum[2], sm[2]);
        // avg

        //atomicAdd(&d_global_pixel_avg[0], (int)sm[0] / blocks);
        //atomicAdd(&d_global_pixel_avg[1], (int)sm[1] / blocks);
        //atomicAdd(&d_global_pixel_avg[2], (int)sm[2] / blocks);

        //d_global_pixel_avg[0] = (unsigned char)((d_global_pixel_avg[0] + (unsigned char)sm[0] / blocks) / blocks);
        //d_global_pixel_avg[1] = (unsigned char)((d_global_pixel_avg[1] + (unsigned char)sm[1] / blocks) / blocks);
        //d_global_pixel_avg[2] = (unsigned char)((d_global_pixel_avg[2] + (unsigned char)sm[2] / blocks) / blocks);
    }
    /*__syncthreads();
    if (blockIdx.x == 0) {
        d_global_pixel_avg[0] += (unsigned char)d_global_pixel_sum[0] / blocks;
        d_global_pixel_avg[1] += (unsigned char)d_global_pixel_sum[1] / blocks;
        d_global_pixel_avg[2] += (unsigned char)d_global_pixel_sum[2] / blocks;
    }*/
    //printf("threadIdx.x %d  idx %d, idx(mod)3 %d\n", threadIdx.x, idx, idx%3);
}

void cuda_stage2(unsigned char* output_global_average) {

    long long int total_pixels = cuda_TILES_X * cuda_TILES_Y * CHANNELS;  // compact pixels

    //printf("total_pixels new %d\n", total_pixels);
    dim3 blocksPerGrid(total_pixels / TILE_SIZE, 1, 1);
    dim3 threadsPerBlock(TILE_SIZE, 1, 1);  // 32

    //compact_mosaic << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_value, d_mosaic_sum, d_global_pixel_sum);

    int blocks = (cuda_TILES_X * cuda_TILES_Y) / TILE_SIZE;
    dim3 blocksPerGrid2(blocks, 1, 1);
    calculate_global_sum << <blocksPerGrid2, threadsPerBlock, TILE_SIZE * CHANNELS * sizeof(unsigned long long) >> > (d_mosaic_value, d_mosaic_sum, d_global_pixel_sum, d_global_pixel_avg, blocks);
    /* wait for all threads to complete */
    //cudaThreadSynchronize();
    //cudaDeviceSynchronize();

    //printf("total_pixels / TILE_SIZE %d\n", total_pixels / TILE_SIZE);
    //printf("(cuda_TILES_X * cuda_TILES_Y) / TILE_SIZE %d\n", (cuda_TILES_X * cuda_TILES_Y) / TILE_SIZE);
    
    
    //unsigned int size = CHANNELS * sizeof(unsigned long long);
    /* allocate the host memory */
    //unsigned long long* h_global_pixel_sum;
    //h_global_pixel_sum = (unsigned long long*)malloc(CHANNELS * sizeof(unsigned long long));
    //cudaMemcpy(h_global_pixel_sum, d_global_pixel_sum, CHANNELS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    //output_global_average[0] = (unsigned char)(h_global_pixel_sum[0] / (cuda_TILES_X * cuda_TILES_Y));
    //output_global_average[1] = (unsigned char)(h_global_pixel_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
    //output_global_average[1] = (unsigned char)(h_global_pixel_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
    //free(h_global_pixel_sum);

    unsigned long long* h_global_pixel_sum;
    unsigned int size = CHANNELS * sizeof(unsigned long long);
    h_global_pixel_sum = (unsigned long long*)malloc(size);
    cudaMemcpy(h_global_pixel_sum, d_global_pixel_sum, size, cudaMemcpyDeviceToHost);
    
    //output_global_average[0] = (unsigned char)(h_global_pixel_sum[0] / (cuda_TILES_X * cuda_TILES_Y));
    //output_global_average[1] = (unsigned char)(h_global_pixel_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
    //output_global_average[2] = (unsigned char)(h_global_pixel_sum[2] / (cuda_TILES_X * cuda_TILES_Y));
    //free(h_global_pixel_sum);
    // 
    // 
    for (int ch = 0; ch < CHANNELS; ++ch) {
        output_global_average[ch] = (unsigned char)(h_global_pixel_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
    //    //printf("global_average= %d\n", h_global_pixel_sum[ch]);
    }
    free(h_global_pixel_sum);



    //for (int ch = 0; ch < CHANNELS; ++ch) {
        //output_global_average[ch] = (unsigned char)(h_global_pixel_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
        //printf("global_average= %d\n", h_global_pixel_sum[ch]);
    //}
    
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    h_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * CHANNELS * sizeof(unsigned char));
    cudaMemcpy(h_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);
#endif    
}

__global__ void cuda_broadcast(unsigned char* d_output_image_data, unsigned char* d_mosaic_value, unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, unsigned int cuda_input_image_width, unsigned int cuda_input_image_height, unsigned int cuda_input_image_channels) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int global_idx = x + y * blockDim.x * gridDim.x;
    int gbl_pixel_idx = global_idx * CHANNELS;
    int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * CHANNELS;

    d_output_image_data[gbl_pixel_idx + 0] = d_mosaic_value[tile_index + 0];
    d_output_image_data[gbl_pixel_idx + 1] = d_mosaic_value[tile_index + 1];
    d_output_image_data[gbl_pixel_idx + 2] = d_mosaic_value[tile_index + 2];


    /*int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int global_idx = x + y * blockDim.x * gridDim.x;*/
    //int gbl_pixel_idx = global_idx * CHANNELS;
    //int tile_index = blockIdx.y * (gridDim.x-1) + blockIdx.x;
    /*int tile_index = (blockIdx.y * (gridDim.x / CHANNELS) + (int)(blockIdx.x / CHANNELS)) * CHANNELS;
    d_output_image_data[global_idx] = d_mosaic_value[tile_index + (global_idx %3)];*/
    //d_output_image_data[gbl_pixel_idx + 1] = d_mosaic_value[tile_index + 1];
    //d_output_image_data[gbl_pixel_idx + 2] = d_mosaic_value[tile_index + 2];
    //if (blockIdx.x == 23 && blockIdx.y == 7) { //(threadIdx.x == 31 && threadIdx.y == 31) {
        //printf("thread_id (%d,%d) block_id (%d,%d) blockDim (%d,%d) gridDim (%d,%d) thread_x_y (%d,%d) global_idx %2d  tile_index %2d local_idx %d\n",
            //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y, x, y, pixel_idx, local_idx, tile_index);

        //printf("thread_id (%d,%d) block_id (%d,%d) blockDim (%d,%d) gridDim (%d,%d) x,y(%d, %d) tile_index %d global_idx %d \n",
            //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, x, y, gridDim.x, gridDim.y, tile_index, global_idx);
    //}
}
void cuda_stage3() {
    //dim3 blocksPerGrid(cuda_TILES_X * CHANNELS, cuda_TILES_Y);
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 32 x 32
    //dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE, 1, 1);  // 32 x 32
    //dim3 blocksPerGrid((cuda_TILES_X * cuda_TILES_Y) / (TILE_SIZE * TILE_SIZE), 1, 1);

    cuda_broadcast << <blocksPerGrid, threadsPerBlock >> > (d_output_image_data, d_mosaic_value, cuda_TILES_X, cuda_TILES_Y, cuda_input_image_width, cuda_input_image_height, cuda_input_image_channels);
    /* wait for all threads to complete */
    //cudaThreadSynchronize();
    //cudaDeviceSynchronize();
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
    CUDA_CALL(cudaFree(d_global_pixel_sum));
    CUDA_CALL(cudaFree(d_global_pixel_avg));
}
