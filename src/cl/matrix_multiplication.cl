#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 16

__kernel void matrix_multiplication(__global const float *as, __global const float *bs, __global float *cs,
                                    const uint M, const uint K, const uint N) {
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    __local float tile_as[LOCAL_SIZE][LOCAL_SIZE];
    __local float tile_bs[LOCAL_SIZE][LOCAL_SIZE];

    float cell = 0.0F;
    for (uint k = 0; k < K; k += LOCAL_SIZE) {
        tile_as[ly][lx] = (y < M && (k + lx) < K) ? as[y * K + (k + lx)] : 0.0F;
        tile_bs[ly][lx] = (x < N && (k + ly) < K) ? bs[(k + ly) * N + x] : 0.0F;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint p = 0; p < LOCAL_SIZE; ++p) {
            cell += tile_as[ly][p] * tile_bs[p][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[y * N + x] = (x < N && y < M) ? cell : 0.0F;
}
