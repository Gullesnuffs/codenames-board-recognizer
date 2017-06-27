#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	double x, y;
} Point;

void dfs_segmentation(
	int height, int width, const unsigned char* im,
	int minimumArea, int maximumArea,
	void (*callback)(int, int, int, int, void*), void* data);

int fit_grid(int size, Point* points, int gridh, int gridw, double* out_score, Point* out_grid);

#ifdef __cplusplus
}
#endif
