# COM4521/COM6521 Assignment Starting Code 2022

This is the starting code for COM4521/COM6521 assignment. You should read the assignment brief from blackboard for details of this code.


CPU "samples/128x128.jpg" "samples/128x128_cpu_out.png" --bench
CPU "samples/128x394.jpg" "samples/128x394_cpu_out.png" --bench
CPU "samples/256x256.png" "samples/256x256_cpu_out.png" --bench
CPU "samples/1024x1024.png" "samples/1024x1024_cpu_out.png" --bench
CPU "samples/2048x2048.png" "samples/2048x2048_cpu_out.png" --bench
CPU "samples/4096x4096.jpg" "samples/4096x4096_cpu_out.png" --bench
CPU "samples/3456x4096.jpg" "samples/3456x4096_cpu_out.png" --bench
CPU "samples/4096x2048.jpg" "samples/4096x2048_cpu_out.png" --bench
CPU "samples/8196x8196.jpg" "samples/8196x8196_cpu_out.png" --bench


OPENMP "samples/128x128.jpg" "samples/128x128_openmp_out.png" --bench
OPENMP "samples/128x394.jpg" "samples/128x394_openmp_out.png" --bench
OPENMP "samples/256x256.png" "samples/256x256_openmp_out.png" --bench
OPENMP "samples/1024x1024.png" "samples/1024x1024_openmp_out.png" --bench
OPENMP "samples/2048x2048.png" "samples/2048x2048_openmp_out.png" --bench
OPENMP "samples/4096x4096.jpg" "samples/4096x4096_openmp_out.png" --bench
OPENMP "samples/3456x4096.jpg" "samples/3456x4096_openmp_out.png" --bench
OPENMP "samples/4096x2048.jpg" "samples/4096x2048_openmp_out.png" --bench
OPENMP "samples/8196x8196.jpg" "samples/8196x8196_openmp_out.png" --bench

CUDA "samples/128x128.jpg" "samples/128x128_cuda_out.png" --bench
CUDA "samples/128x128.jpg" "samples/128x128_cuda_out.png" --bench
CUDA "samples/128x394.jpg" "samples/128x394_cuda_out.png" --bench
CUDA "samples/256x256.png" "samples/256x256_cuda_out.png" --bench
CUDA "samples/1024x1024.png" "samples/1024x1024_cuda_out.png" --bench
CUDA "samples/2048x2048.png" "samples/2048x2048_cuda_out.png" --bench
CUDA "samples/4096x4096.jpg" "samples/4096x4096_cuda_out.png" --bench
CUDA "samples/3456x4096.jpg" "samples/3456x4096_cuda_out.png" --bench
CUDA "samples/4096x2048.jpg" "samples/4096x2048_cuda_out.png" --bench
CUDA "samples/8196x8196.jpg" "samples/8196x8196_cuda_out.png" --bench


OPENMP "U:\pramodpots\COM6521\photo-mosaic\x64\samples\4096x4096.jpg" "U:\pramodpots\COM6521\photo-mosaic\x64\samples\4096x4096_cuda_out.png"


OPENMP "samples/64x64.jpg" "samples/64x64_out.png" --bench
OPENMP "samples/79x91.jpg" "samples/79x91_out.png" --bench
OPENMP "samples/3096x5000.jpg" "samples/3096x5000_out.png" --bench
OPENMP "samples/2345x5678.jpg" "samples/2345x5678_out.png" --bench
OPENMP "samples/6879x2974.jpg" "samples/6879x2974_out.png" --bench
OPENMP "samples/69x4096.jpg" "samples/69x4096_out.png" --bench
OPENMP "samples/4096x69.jpg" "samples/4096x69_out.png" --bench

CUDA "samples/64x64.jpg" "samples/64x64_cuout.png" --bench
CUDA "samples/79x91.jpg" "samples/79x91_cuout.png" --bench
CUDA "samples/3096x5000.jpg" "samples/3096x5000_cuout.png" --bench
CUDA "samples/2345x5678.jpg" "samples/2345x5678_cuout.png" --bench
CUDA "samples/6879x2974.jpg" "samples/6879x2974_cuout.png" --bench
CUDA "samples/69x4096.jpg" "samples/69x4096_cuda_out.png" --bench
CUDA "samples/4096x69.jpg" "samples/4096x69_cuda_out.png" --bench