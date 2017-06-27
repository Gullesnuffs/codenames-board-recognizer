from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source("_grid", """
#include "grid.h"
    """,
    libraries=[], extra_objects=['grid.o'], extra_link_args=['-lstdc++'])

ffibuilder.cdef("""
typedef struct {
	double x, y;
} Point;

extern "Python" void dfs_segmentation_append(int, int, int, int, void*);

void dfs_segmentation(
    int height, int width, const unsigned char* im,
    int minimumArea, int maximumArea,
    void (*callback)(int, int, int, int, void*), void* data);

int fit_grid(int size, Point* points, int gridh, int gridw, double* out_score, Point* out_grid);
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
