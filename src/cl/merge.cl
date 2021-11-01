#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 256

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
