#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 256

// http://www.bealto.com/gpu-sorting_parallel-merge-local.html
__kernel void merge_local(__global const float *in, __global float *out, int n) {
    __local float temp[LOCAL_SIZE];
    const int id = get_global_id(0);
    const int i = get_local_id(0);
    temp[i] = id < n ? in[id] : 2147483648.0F;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int sorted = 1; sorted != LOCAL_SIZE; sorted *= 2) {
        const float i_data = temp[i];
        const int i_remainder = i & (sorted - 1);
        const int sibling = (i - i_remainder) ^ sorted;
        int pos = 0;
        for (int inc = sorted; inc != 0; inc /= 2) {
            const int j = sibling + pos + inc - 1;
            const float j_data = temp[j];
            if (j_data < i_data || (j_data == i_data && j < i)) {
                pos = min(pos + inc, sorted);
            }
        }
        const int bits = 2 * sorted - 1;
        const int dest = ((i_remainder + pos) & bits) | (i & ~bits);
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[dest] = i_data;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id < n) {
        out[id] = temp[i];
    }
}

__kernel void merge(__global const float *in, __global float *out, int sorted, int n) {
    const int id = get_global_id(0);
    if (id >= n) {
        return;
    }
    const int first_offset = (id / (2 * sorted)) * 2 * sorted;
    const int second_offset = first_offset + sorted;
    if (second_offset >= n) {
        out[id] = in[id];
        return;
    }
    const int is_first = id < second_offset;
    int low = is_first ? second_offset : first_offset;
    int up = is_first ? min(second_offset + sorted, n) : second_offset;
    const float item = in[id];
    while (low < up) {
        int m = (low + up) / 2;
        if (in[m] < item) {
            low = m + 1;
        } else {
            up = m;
        }
    }
    up = low;
    while (up < second_offset && in[up] == item) {
        ++up;
    }
    out[id + (is_first ? low : up) - second_offset] = item;
}
