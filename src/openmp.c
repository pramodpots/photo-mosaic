#include "openmp.h"
#include "helper.h"

///
/// Algorithm storage
///
Image omp_input_image;
Image omp_output_image;
unsigned int omp_TILES_X, omp_TILES_Y;
unsigned long long* omp_mosaic_sum;
unsigned char* omp_mosaic_value;

void openmp_begin(const Image *input_image) {
    omp_TILES_X = input_image->width / TILE_SIZE;
    omp_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    omp_mosaic_sum = (unsigned long long*)malloc(omp_TILES_X * omp_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    omp_mosaic_value = (unsigned char*)malloc(omp_TILES_X * omp_TILES_Y * input_image->channels * sizeof(unsigned char));
    
    // Allocate copy of input image
    omp_input_image = *input_image;
    omp_input_image.data = (unsigned char *)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(omp_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    omp_output_image = *input_image;
    omp_output_image.data = (unsigned char *)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
}
void openmp_stage1() {
    // Reset sum memory to 0
    memset(omp_mosaic_sum, 0, omp_TILES_X * omp_TILES_Y * omp_input_image.channels * sizeof(unsigned long long));
    // Sum pixel data within each tile
    for (unsigned int t_x = 0; t_x < omp_TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < omp_TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * omp_TILES_X + t_x) * omp_input_image.channels;
            const unsigned int tile_offset = (t_y * omp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * omp_input_image.channels;
            // For each pixel within the tile
            for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    // For each colour channel
                    const unsigned int pixel_offset = (p_y * omp_input_image.width + p_x) * omp_input_image.channels;
                    for (int ch = 0; ch < omp_input_image.channels; ++ch) {
                        // Load pixel
                        const unsigned char pixel = omp_input_image.data[tile_offset + pixel_offset + ch];
                        omp_mosaic_sum[tile_index + ch] += pixel;
                    }
                }
            }
        }
    }
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // validate_tile_sum(&input_image, mosaic_sum);
#endif
}
void openmp_stage2(unsigned char* output_global_average) {
    // Calculate the average of each tile, and sum these to produce a whole image average.
    unsigned long long whole_image_sum[4] = {0, 0, 0, 0};  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    for (unsigned int t = 0; t < omp_TILES_X * omp_TILES_Y; ++t) {
        for (int ch = 0; ch < omp_input_image.channels; ++ch) {
            omp_mosaic_value[t * omp_input_image.channels + ch] = (unsigned char)(omp_mosaic_sum[t * omp_input_image.channels + ch] / TILE_PIXELS);  // Integer division is fine here
            whole_image_sum[ch] += omp_mosaic_value[t * omp_input_image.channels + ch];
        }
    }
    // Reduce the whole image sum to whole image average for the return value
    for (int ch = 0; ch < omp_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (omp_TILES_X * omp_TILES_Y));
    }
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // validate_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, mosaic_value, output_global_average);
#endif    
}
void openmp_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    // For each tile
    for (unsigned int t_x = 0; t_x < omp_TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < omp_TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * omp_TILES_X + t_x) * omp_input_image.channels;
            const unsigned int tile_offset = (t_y * omp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * omp_input_image.channels;
            
            // For each pixel within the tile
            for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    const unsigned int pixel_offset = (p_y * omp_input_image.width + p_x) * omp_input_image.channels;
                    // Copy whole pixel
                    memcpy(omp_output_image.data + tile_offset + pixel_offset, omp_mosaic_value + tile_index, omp_input_image.channels);
                }
            }
        }
    }
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // validate_broadcast(&input_image, mosaic_value, &output_image);
#endif    
}
void openmp_end(Image *output_image) {
    
    // Store return value
    output_image->width = omp_output_image.width;
    output_image->height = omp_output_image.height;
    output_image->channels = omp_output_image.channels;
    memcpy(output_image->data, omp_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(omp_output_image.data);
    free(omp_input_image.data);
    free(omp_mosaic_value);
    free(omp_mosaic_sum);
}