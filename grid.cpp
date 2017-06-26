#include <vector>
#include <algorithm>
#include <iostream>
#include <tuple>
#include "grid.h"
using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
typedef pair<int, int> pii;

extern "C"
void dfs_segmentation(
	int height, int width, const unsigned char* im,
	int minimumArea, int maximumArea,
    void (*callback)(int, int, int, int, void*), void* data)
{
    // Do DFS from each black pixel to find regions of black pixels
	vector<vector<bool>> visited(height, vector<bool>(width));

	rep(i, 0, height)
        visited[i][0] = visited[i][width-1] = true;
	rep(j, 0, width)
        visited[0][j] = visited[height-1][j] = true;

    const int DX[4] = {1, 0, -1, 0};
    const int DY[4] = {0, 1, 0, -1};
	vector<pii> q;
	rep(x, 0, width) rep(y, 0, height) {
		if (visited[y][x]) continue;
		if (im[y * width + x] == 255) continue;
		q.emplace_back(x, y);
		int minX = x, maxX = x;
		int minY = y, maxY = y;
		visited[y][x] = true;
		while (!q.empty()) {
			int cx, cy;
			tie(cx, cy) = q.back();
			q.pop_back();
			minX = min(minX, cx);
			maxX = max(maxX, cx);
			minY = min(minY, cy);
			maxY = max(maxY, cy);
			rep(di, 0, 4) {
				int nx = cx + DX[di];
				int ny = cy + DY[di];
				if (visited[ny][nx]) continue;
				if (im[ny * width + nx] == 255) continue;
				visited[ny][nx] = true;
				q.emplace_back(nx, ny);
			}
		}
		int regionWidth = maxX - minX + 1;
		int regionHeight = maxY - minY + 1;
		int bbarea = regionWidth * regionHeight;
		if (bbarea < minimumArea || bbarea > maximumArea) continue;
		callback(minX, minY, maxX, maxY, data);
	}
}
