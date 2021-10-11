#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 16

__kernel void matrix_transpose(__global const float *input, __global float *output, const uint M, const uint K) {
    __local float tile[LOCAL_SIZE][LOCAL_SIZE];
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint tile_id = (lx + ly) & 0xF;
    if (x < K && y < M) {
        tile[ly][tile_id] = input[y * K + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x < K && y < M) {
        output[(x - lx + ly) * M + y - ly + lx] = tile[lx][tile_id];
    }
}
