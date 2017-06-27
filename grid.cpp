#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <array>
#include <cmath>
#include "grid.h"
using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define trav(x, v) for (auto &x : v)
typedef pair<int, int> pii;
typedef vector<int> vi;

static const double inf = 1e9;

Point operator+(Point a, Point b) {
	return {a.x + b.x, a.y + b.y};
}

Point operator-(Point a, Point b) {
	return {a.x - b.x, a.y - b.y};
}

Point operator*(Point a, int n) {
	return {a.x * n, a.y * n};
}

double dist2(Point p, Point q) {
	Point d = p - q;
	return d.x*d.x + d.y*d.y;
}

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


double fit_grid2(
		const vector<Point>& points,
		int gridh, int gridw,
		Point topleft, Point topright, Point botleft, Point botright,
		vector<Point>* out)
{
	auto gridp = [&](int i, int j) -> Point {
		Point a = topleft + (topright - topleft) * j;
		Point b = botleft + (botright - botleft) * j;
		return b * i + a * (1 - i);
	};

	const double RAD_2 = 20 * 20;
	const double INV_RAD_2 = 1 / RAD_2;

	int npoints = (int)points.size();

	assert(gridh >= 2 && gridw >= 2);
	int minh = -gridh + 2, maxh = gridh;
	int minw = -gridw + 2, maxw = gridw;
	vector<vector<double>> scores(maxh - minh, vector<double>(maxw - minw));
	vector<vector<int>> assignments(maxh - minh, vector<int>(maxw - minw));

	rep(i, minh, maxh) rep(j, minw, maxw) {
		Point p = gridp(i, j);
		double best_d2 = inf;
		int bestk = -1;
		rep(k, 0, npoints) {
			double d2 = dist2(p, points[k]);
			if (d2 < best_d2) {
				best_d2 = d2;
				bestk = k;
			}
		}
		double score = max(1 - best_d2 * INV_RAD_2, 0.);
		// double rat = best_dis2 / RAD_2;
		// double score = (rad < 0.5 ? 1 - rat : exp(-rat) * 0.82436);
		// double score = (best_dis2 < RAD_2 ? 1 : 0);
		scores[i - minh][j - minw] = score;
		assignments[i - minh][j - minw] = bestk;
	}

	tuple<double, int, int> bestsc{-1, 0, 0};
	vi uses(npoints);
	rep(i, minh, 1) rep(j, minw, 1) {
		double sc = 0;
		uses.assign(npoints, 0);
		rep(ik, 0, gridh) rep(jk, 0, gridw) {
			double s = scores[i + ik - minh][j + jk - minw];
			int as = assignments[i + ik - minh][j + jk - minw];
			sc += s / ++uses[as];
		}
		bestsc = max(bestsc, make_tuple(sc, i, j));
	}

	double score;
	int bi, bj;
	tie(score, bi, bj) = bestsc;
	assert(score != -1);
	rep(i, 0, gridh) rep(j, 0, gridw)
		(*out)[i * gridw + j] = gridp(bi + i, bj + j);
	return score;
}


extern "C"
int fit_grid(
		int size, Point* inPoints,
		int gridh, int gridw,
		double* out_score, Point* out_grid)
{
	const vector<Point> points(inPoints, inPoints + size);

	double bestscore = -1;
	vector<Point> bestgrid(gridh * gridw), tempgrid(gridh * gridw);
	array<Point, 4> bestcorners{};
	double min_dist = 10;

	trav(topleft, points) {
		// We'll try to find a reasonable 2x2 subsquare (topleft, topright,
		// botleft, botright) of the grid.
		// Step 1: Find the nearest points to the right and down of this one.
		double trdist = inf, bldist = inf;
		Point topright{}, botleft{};
		trav(pt2, points) {
			double dx = pt2.x - topleft.x;
			double dy = pt2.y - topleft.y;
			if (min_dist < dx && dx < trdist && abs(dy) < dx * 0.6) {
				trdist = dx;
				topright = pt2;
			}
			if (min_dist < dy && dy < bldist && abs(dx) < dy * 0.6) {
				bldist = dy;
				botleft = pt2;
			}
		}

		// Make sure we have found both points
		if (trdist == inf || bldist == inf)
			continue;

		// Find the two points right-down and down-right of this one.
		double drdist = inf, rddist = inf;
		Point downright{}, rightdown{};
		trav(pt2, points) {
			double dx = pt2.x - botleft.x;
			double dy = pt2.y - botleft.y;
			if (min_dist < dx && dx < drdist && abs(dy) < dx * 0.6) {
				drdist = dx;
				downright = pt2;
			}
			dx = pt2.x - topright.x;
			dy = pt2.y - topright.y;
			if (min_dist < dy && dy < rddist && abs(dx) < dy * 0.6) {
				rddist = dy;
				rightdown = pt2;
			}
		}

		// Make sure the points exist and coincide, so we actually get a quadrilateral.
		if (downright.x != rightdown.x || downright.y != rightdown.y || drdist == inf || rddist == inf)
			continue;
		Point botright = rightdown;

		// Try basing our grid on those four points, and see what fits.
		double score = fit_grid2(points, gridh, gridw,
				topleft, topright, botleft, botright, &tempgrid);
		if (score > bestscore) {
			bestscore = score;
			bestgrid.swap(tempgrid);
			bestcorners = {{topleft, topright, botleft, botright}};
		}
	}

	if (bestscore < 0)
		return 0;

	// Hill climb a bit to improve the solution
	for (double delta = 2.0; delta > 0.2; delta /= 2) {
		// Try to offset all corners by -1, 0 or 1 times the current delta, in
		// both X and Y direction.
		rep(dir, 0, 2) {
			double Point::*field = (dir ? &Point::x : &Point::y);
			int d[4];
			for (d[0] = -1; d[0] <= 1; ++d[0])
			for (d[1] = -1; d[1] <= 1; ++d[1])
			for (d[2] = -1; d[2] <= 1; ++d[2])
			for (d[3] = -1; d[3] <= 1; ++d[3]) {
				auto c = bestcorners;
				c[0].*field += d[0] * delta;
				c[1].*field += d[1] * delta;
				c[2].*field += d[2] * delta;
				c[3].*field += d[3] * delta;
				// Check if offsetting the corners like this produces a better fit
				double score = fit_grid2(points, gridh, gridw,
						c[0], c[1], c[2], c[3], &tempgrid);
				if (score > bestscore) {
					bestscore = score;
					bestgrid.swap(tempgrid);
					bestcorners = c;
				}
			}
		}
	}

	*out_score = bestscore;
	copy(bestgrid.begin(), bestgrid.end(), out_grid);
	return 1;
}
