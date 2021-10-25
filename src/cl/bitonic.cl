#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 256

__kernel void bitonic_small(__global float *as, uint stage_size, uint size, uint n) {
    uint local_id = get_local_id(0);
    uint id = get_global_id(0);
    __local float local_as[LOCAL_SIZE];
    if (id < n) {
        local_as[local_id] = as[id];
    }
    uint flag = id % (2 * size) < size;
    barrier(CLK_LOCAL_MEM_FENCE);
    while (stage_size > 0) {
        if (id + stage_size < n && id % (2 * stage_size) < stage_size) {
            float a = local_as[local_id];
            float b = local_as[local_id + stage_size];
            if ((a > b) == flag) {
                local_as[local_id] = b;
                local_as[local_id + stage_size] = a;
            }
        }
        stage_size /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id < n) {
        as[id] = local_as[local_id];
    }
}

__kernel void bitonic_large(__global float *as, uint stage_size, uint size, uint n) {
    uint id = get_global_id(0);
    if (id + stage_size < n && id % (2 * stage_size) < stage_size) {
        float a = as[id];
        float b = as[id + stage_size];
        if ((a > b) == (id % (2 * size) < size)) {
            as[id] = b;
            as[id + stage_size] = a;
        }
    }
}
