from __future__ import print_function
import cv2
import sys
from termcolor import colored


SIZE = 5


def show(im):
    from PIL import Image
    # cv2.imshow('image', im)
    Image.fromarray(im).show()


def circ_dilation(dilation):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2 * dilation + 1, 2 * dilation + 1),
                                     (dilation, dilation))


def dfs_segmentation(im, minimumArea, maximumArea):
    # Do DFS from each black pixel to find regions of black pixels
    height, width = im.shape
    visited = [[False] * width for _ in range(height)]

    for i in range(height):
        visited[i][0] = visited[i][width-1] = True
    for j in range(width):
        visited[0][j] = visited[height-1][j] = True

    DX = [1, 0, -1, 0]
    DY = [0, 1, 0, -1]
    q = []
    for x in range(width):
        for y in range(height):
            if visited[y][x]:
                continue
            if im[y][x] == 255:
                continue
            area = 0
            q.append((x, y))
            minX = maxX = x
            minY = maxY = y
            visited[y][x] = True
            while len(q) > 0:
                cx, cy = q.pop()
                area += 1
                minX = min(minX, cx)
                maxX = max(maxX, cx)
                minY = min(minY, cy)
                maxY = max(maxY, cy)
                for di in range(4):
                    nx = cx + DX[di]
                    ny = cy + DY[di]
                    if visited[ny][nx]:
                        continue
                    if im[ny][nx] == 255:
                        continue
                    visited[ny][nx] = True
                    q.append((nx, ny))
            regionWidth = maxX - minX + 1
            regionHeight = maxY - minY + 1
            bbarea = regionWidth * regionHeight
            if bbarea < minimumArea or bbarea > maximumArea:
                continue
            box = (minX, minY, maxX, maxY)
            # print("found box " + str(box) + " " + str(bbarea))
            # box = clamp_aabb(expand_aabb(box, 3), width, height)
            # averageR = totR/totPixels
            # averageG = totG/totPixels
            # averageB = totB/totPixels
            yield box
            # yield (averageR, averageG, averageB, box)


def fit_grid(points):
    inf = 10**9
    bestscore = -1
    bestgrid = None
    for pt in points:
        # Find a reasonable subsquare (pt, bestr, bestd, bestrd)
        bestrx = bestdy = inf
        bestr = bestd = None
        for pt2 in points:
            dy = pt2[0] - pt[0]
            dx = pt2[1] - pt[1]
            if 0 < dx < bestrx and abs(dy) < dx * 0.6:
                bestrx = dx
                bestr = pt2
            if 0 < dy < bestdy and abs(dx) < dy * 0.6:
                bestdy = dy
                bestd = pt2
        if not bestr or not bestd:
            continue
        bestrx = bestdy = inf
        bestdr = bestrd = None
        for pt2 in points:
            dy = pt2[0] - bestd[0]
            dx = pt2[1] - bestd[1]
            if 0 < dx < bestrx and abs(dy) < dx * 0.6:
                bestrx = dx
                bestdr = pt2
            dy = pt2[0] - bestr[0]
            dx = pt2[1] - bestr[1]
            if 0 < dy < bestdy and abs(dx) < dy * 0.6:
                bestdy = dy
                bestrd = pt2
        if bestdr != bestrd or not bestrd:
            continue

        # Try basing our grid on those four points, and see what fits.
        topleft = complex(pt[0], pt[1])
        topright = complex(bestr[0], bestr[1])
        botleft = complex(bestd[0], bestd[1])
        botright = complex(bestrd[0], bestrd[1])
        def gridp(i, j):
            a = topleft + j * (topright - topleft)
            b = botleft + j * (botright - botleft)
            return b * i + a * (1 - i)
        # assert gridp(0,0) == topleft
        # assert gridp(1,0) == botleft
        # assert gridp(0,1) == topright
        # assert gridp(1,1) == botright

        def top(c):
            return (int(c.imag), int(c.real))

        match_dis2 = 10**2
        minr, maxr = -SIZE+2, SIZE
        scores = [[0] * (maxr - minr) for _ in range(maxr - minr)]
        for i in range(minr, maxr):
            for j in range(minr, maxr):
                p = gridp(i, j)
                best_dis2 = inf
                for k in range(len(points)):
                    q = complex(points[k][0], points[k][1])
                    dis = p - q
                    best_dis2 = min(best_dis2, dis.real**2 + dis.imag**2)
                scores[i - minr][j - minr] = 1 if best_dis2 < match_dis2 else 0

        bestsc = (-1, 0, 0)
        for i in range(minr, 1):
            for j in range(minr, 1):
                sc = 0
                for ik in range(SIZE):
                    for jk in range(SIZE):
                        sc += scores[i + ik - minr][j + jk - minr]
                bestsc = max(bestsc, (sc, i, j))

        score = bestsc[0]
        assert score != -1
        bi = bestsc[1]
        bj = bestsc[2]
        if score > bestscore:
            bestscore = score
            bestgrid = [[top(gridp(bi + i, bj + j)) for i in range(SIZE)] for j in range(SIZE)]

    if bestscore <= SIZE*SIZE//2:
        return None
    return bestgrid


"""
def getcolor(r, g, b):
  if r > g*2 and r > b*2 and r > 70:
    return "r"
  elif b > r*2 and b > g*0.8 and b > 60:
    return "b"
  elif r > 40 and g > 40 and b > 40:
    return "c"
  else:
    return "a"
"""


def find_words(fname):
    im = cv2.imread(fname)

    desiredSize = 512
    scale = desiredSize / max(im.shape[0], im.shape[1])
    newSize = (round(im.shape[1] * scale), round(im.shape[0] * scale))
    im = cv2.resize(im, newSize)

    im = cv2.GaussianBlur(im, (3, 3), 0)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10)

    gray = cv2.erode(gray, circ_dilation(1))
    gray = cv2.dilate(gray, circ_dilation(1))

    # show(gray)
    # contIm, contours, hierarchy = cv2.findContours(gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # show(contIm)

    areas = list(dfs_segmentation(gray, 80, 512**2 // 20))
    points = [((ar[1] + ar[3]) // 2, (ar[0] + ar[2]) // 2) for ar in areas]
    print(len(points))

    grid = fit_grid(points)

    print(grid)

    height, width = gray.shape
    for row in grid:
        for co in row:
            x, y = co
            x -= 5
            y -= 5
            for i in range(11):
                for j in range(11):
                    gray[max(min(y+i, height-1), 0)][max(min(x+j, width-1), 0)] = 128

    show(gray)
    exit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 secret.py image")
        exit(1)
    foundWords, grid = find_words(sys.argv[1])
    print("Found " + str(len(foundWords)) + " candidates!")
    for i in range(5):
        line = ""
        for j in range(5):
            if grid[i][j] == "b":
                line += colored('■ ', 'blue')
            elif grid[i][j] == "r":
                line += colored('■ ', 'red')
            elif grid[i][j] == "c":
                line += colored('■ ', 'grey')
            elif grid[i][j] == "a":
                line += colored('■ ', 'magenta')
            else:
                line += "u "
        print(line)
