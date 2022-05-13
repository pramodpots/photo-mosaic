# COM4521/COM6521 Assignment Starting Code 2022

This is the starting code for COM4521/COM6521 assignment. You should read the assignment brief from blackboard for details of this code.


CPU "samples/256x256.png" "samples/256x256_cpu_out.png" --bench
CPU "samples/1024x1024.png" "samples/1024x1024_cpu_out.png" --bench
CPU "samples/2048x2048.png" "samples/2048x2048_cpu_out.png" --bench
CPU "samples/4096x4096.jpg" "samples/4096x4096_cpu_out.png" --bench


OPENMP "samples/256x256.png" "samples/256x256_openmp_out.png" --bench
OPENMP "samples/1024x1024.png" "samples/1024x1024_openmp_out.png" --bench
OPENMP "samples/2048x2048.png" "samples/2048x2048_openmp_out.png" --bench
OPENMP "samples/4096x4096.jpg" "samples/4096x4096_openmp_out.png" --bench


CUDA "samples/256x256.png" "samples/256x256_cuda_out.png" --bench
CUDA "samples/1024x1024.png" "samples/1024x1024_cuda_out.png" --bench
CUDA "samples/2048x2048.png" "samples/2048x2048_cuda_out.png" --bench
CUDA "samples/4096x4096.jpg" "samples/4096x4096_cuda_out.png" --bench
