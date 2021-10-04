#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 256

__kernel void sum(__global const uint *as, __global uint *result) {
    const uint globalId = get_global_id(0);
    const uint localId = get_local_id(0);
    __local uint localAs[LOCAL_SIZE];
    localAs[localId] = as[globalId];
    for (uint offset = LOCAL_SIZE / 2; offset != 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < offset) {
            localAs[localId] += localAs[localId + offset];
        }
    }
    if (localId == 0) {
        atomic_add(result, localAs[0]);
    }
}
