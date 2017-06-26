#ifdef __cplusplus
extern "C" {
#endif

void dfs_segmentation(
	int height, int width, const unsigned char* im,
	int minimumArea, int maximumArea,
    void (*callback)(int, int, int, int, void*), void* data);

#ifdef __cplusplus
}
#endif
