from __future__ import print_function
import cv2
import sys
import math
import numpy as np
from termcolor import colored


SIZE = 5
INF = 10**9


def show(im):
    from PIL import Image
    # cv2.imshow('image', im)
    Image.fromarray(im).show()


def circ_dilation(rad):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1), (rad, rad))


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
            yield box


def fit_grid2(points, topleft, topright, botleft, botright):
    def gridp(i, j):
        a = topleft + j * (topright - topleft)
        b = botleft + j * (botright - botleft)
        return b * i + a * (1 - i)

    def top(c):
        return (int(c.imag), int(c.real))

    match_dis2 = 20**2
    minr, maxr = -SIZE+2, SIZE
    scores = [[(0, -1)] * (maxr - minr) for _ in range(maxr - minr)]
    for i in range(minr, maxr):
        for j in range(minr, maxr):
            p = gridp(i, j)
            best_dis2 = INF
            bestk = -1
            for k in range(len(points)):
                q = complex(points[k][0], points[k][1])
                dis = p - q
                dis2 = dis.real**2 + dis.imag**2
                if dis2 < best_dis2:
                    best_dis2 = dis2
                    bestk = k
            score = max(1 - best_dis2 / match_dis2, 0)
            # rat = best_dis2 / match_dis2
            # score = 1 - rat if rat < 0.5 else math.exp(-rat) * 0.82436
            # score = 1 if best_dis2 < match_dis2 else 0
            scores[i - minr][j - minr] = (score, bestk)

    bestsc = (-1, 0, 0)
    for i in range(minr, 1):
        for j in range(minr, 1):
            sc = 0
            used = [0] * len(points)
            for ik in range(SIZE):
                for jk in range(SIZE):
                    s = scores[i + ik - minr][j + jk - minr]
                    used[s[1]] += 1
                    u = used[s[1]]
                    sc += s[0] / u
            bestsc = max(bestsc, (sc, i, j))

    score = bestsc[0]
    assert score != -1
    bi = bestsc[1]
    bj = bestsc[2]
    grid = [[top(gridp(bi + i, bj + j)) for j in range(SIZE)] for i in range(SIZE)]
    return (score, grid)


def fit_grid(points):
    bestscore = -1
    bestgrid = None
    bestcorners = None
    min_dist = 20
    for pt in points:
        # Find a reasonable subsquare (pt, bestr, bestd, bestrd)
        bestrx = bestdy = INF
        bestr = bestd = None
        for pt2 in points:
            dy = pt2[0] - pt[0]
            dx = pt2[1] - pt[1]
            if min_dist < dx < bestrx and abs(dy) < dx * 0.6:
                bestrx = dx
                bestr = pt2
            if min_dist < dy < bestdy and abs(dx) < dy * 0.6:
                bestdy = dy
                bestd = pt2
        if not bestr or not bestd:
            continue
        bestrx = bestdy = INF
        bestdr = bestrd = None
        for pt2 in points:
            dy = pt2[0] - bestd[0]
            dx = pt2[1] - bestd[1]
            if min_dist < dx < bestrx and abs(dy) < dx * 0.6:
                bestrx = dx
                bestdr = pt2
            dy = pt2[0] - bestr[0]
            dx = pt2[1] - bestr[1]
            if min_dist < dy < bestdy and abs(dx) < dy * 0.6:
                bestdy = dy
                bestrd = pt2
        if bestdr != bestrd or not bestrd:
            continue

        # Try basing our grid on those four points, and see what fits
        topleft = complex(pt[0], pt[1])
        topright = complex(bestr[0], bestr[1])
        botleft = complex(bestd[0], bestd[1])
        botright = complex(bestrd[0], bestrd[1])
        score, grid = fit_grid2(points, topleft, topright, botleft, botright)
        if score > bestscore:
            bestscore = score
            bestgrid = grid
            bestcorners = (topleft, topright, botleft, botright)

    if not bestgrid:
        return None

    # Hill climb a bit to improve the solution
    for delta in [2., 1., .5, .25]:
        DIRY = [complex(delta * i, 0) for i in range(-1, 2)]
        DIRX = [complex(0, delta * i) for i in range(-1, 2)]
        for dira in DIRY:
            for dirb in DIRY:
                for dirc in DIRY:
                    for dird in DIRY:
                        c = bestcorners
                        ca, cb, cc, cd = c[0] + dira, c[1] + dirb, c[2] + dirc, c[3] + dird
                        score, grid = fit_grid2(points, ca, cb, cc, cd)
                        if score > bestscore:
                            bestscore = score
                            bestgrid = grid
                            bestcorners = (ca, cb, cc, cd)
        for dira in DIRX:
            for dirb in DIRX:
                for dirc in DIRX:
                    for dird in DIRX:
                        c = bestcorners
                        ca, cb, cc, cd = c[0] + dira, c[1] + dirb, c[2] + dirc, c[3] + dird
                        score, grid = fit_grid2(points, ca, cb, cc, cd)
                        if score > bestscore:
                            bestscore = score
                            bestgrid = grid
                            bestcorners = (ca, cb, cc, cd)


    # if bestscore <= SIZE*SIZE//2: return None
    print(bestscore)
    return bestgrid


def getcolor(col):
    b, g, r = col
    if r > g*2 and r > b*2 and r > 70:
        return "r"
    elif b > r*2 and b > g*0.8 and b > 60:
        return "b"
    elif r > 40 and g > 40 and b > 40:
        return "c"
    else:
        return "a"


def fancy_thresholding(im):
    r, g, b = cv2.split(im)
    c = np.maximum(r, np.maximum(g, b))
    cv2.imwrite('t1.png', c)
    # c = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    c = cv2.GaussianBlur(c, (0, 0), 2)
    cv2.imwrite('t2.png', c)
    mid = cv2.GaussianBlur(c, (0, 0), 8)
    c = (3 * (c.astype(np.float32) - mid.astype(np.float32)) + 128).clip(0, 255).astype(np.uint8)
    _, c = cv2.threshold(c, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite('t3.png', c)

    dilation = 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilation + 1, 2 * dilation + 1),
                                      (dilation, dilation))
    c = cv2.erode(c, element)
    cv2.imwrite('t4.png', c)

    # des = cv2.bitwise_not(gray)

    # Fill in small contours
    _, contour, _ = cv2.findContours(c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        if cv2.contourArea(cnt) < 20000:
            cv2.drawContours(c, [cnt], 0, 255, -1)

    cv2.imwrite('t5.png', c)
    dilation = 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilation + 1, 2 * dilation + 1),
                                      (dilation, dilation))
    c = cv2.erode(c, element)
    cv2.imwrite('t6.png', c)

    # gray = cv2.resize(gray, (newSize[0]//2, newSize[1]//2))
    c = 255 - c
    return c


def some_other_thresholding(im):
    im = cv2.GaussianBlur(im, (3, 3), 0)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10)

    gray = cv2.erode(gray, circ_dilation(1))
    gray = cv2.dilate(gray, circ_dilation(1))
    return gray


def resize(im, desiredWidth):
  scale = desiredWidth / max(im.shape[0], im.shape[1])
  newSize = (round(im.shape[1] * scale), round(im.shape[0] * scale))
  im = cv2.resize(im, newSize)
  return im


def find_grid(fname):
    im = cv2.imread(fname)
    if im is None:
        raise Exception("Could not read file: " + fname)

    # im = resize(im, 512)
    # gray = some_other_thresholding(im)

    im = resize(im, 1024)
    gray = fancy_thresholding(im)
    gray = resize(gray, 512)

    # show(gray)
    # contIm, contours, hierarchy = cv2.findContours(gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # show(contIm)

    height, width = gray.shape

    areas = list(dfs_segmentation(gray, 80, 512**2 // 20))
    points = [((ar[1] + ar[3]) / 2, (ar[0] + ar[2]) / 2) for ar in areas]
    print(len(points))

    grid = fit_grid(points)

    mat = []
    for row in grid:
        mrow = ''
        for co in row:
            x, y = co
            x = max(min(x, width-1), 0)
            y = max(min(y, height-1), 0)
            mrow += getcolor(im[y][x])
        mat.append(mrow)

    for row in grid:
        for co in row:
            x, y = co
            x -= 5
            y -= 5
            for i in range(11):
                for j in range(11):
                    gray[max(min(y+i, height-1), 0)][max(min(x+j, width-1), 0)] = 128

    show(gray)
    return mat


def colorize(c):
    if c == "b": return colored('■ ', 'blue')
    if c == "r": return colored('■ ', 'red')
    if c == "c": return colored('■ ', 'grey')
    if c == "a": return colored('■ ', 'cyan')
    return "u "


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grid.py image")
        exit(1)
    grid = find_grid(sys.argv[1])
    for row in grid:
        print(''.join(colorize(c) for c in row))
