from __future__ import print_function
import cv2
import sys
if __name__ == '__main__':
    from _grid import ffi, lib
else:
    from ._grid import ffi, lib
import numpy as np
from termcolor import colored
import math


SIZE = 5
INF = 10**9


def show(im):
    from PIL import Image
    print(im.shape)
    # Fix BGR -> RGB
    if im.shape[-1] == 3:
        b, g, r = cv2.split(im)
        im = cv2.merge([r, g, b])
    Image.fromarray(im).show()


def circ_dilation(rad):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1), (rad, rad))


def dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return (dx*dx + dy*dy) ** 0.5


@ffi.def_extern()
def dfs_segmentation_append(minX, minY, maxX, maxY, data):
    li = ffi.from_handle(data)
    li.append((minX, minY, maxX, maxY))


def dfs_segmentation(im, minimumArea, maximumArea):
    height, width = im.shape
    assert im.strides == (width, 1)
    imdata = ffi.cast("unsigned char *", im.ctypes.data)

    res = []
    lib.dfs_segmentation(
        height, width, imdata,
        minimumArea, maximumArea,
        lib.dfs_segmentation_append, ffi.new_handle(res))
    return res


def fit_grid(points):
    ps = ffi.new('Point[]', len(points))
    for i in range(len(points)):
        ps[i].y, ps[i].x = points[i]
    score_output = ffi.new('double*')
    output = ffi.new('Point[]', SIZE*SIZE)
    if not lib.fit_grid(len(points), ps, 5, 5, score_output, output):
        return None
    score = score_output[0]
    res = [[(0,0)] * SIZE for _ in range(SIZE)]
    for i in range(SIZE):
        for j in range(SIZE):
            p = output[i * SIZE + j]
            res[i][j] = (p.x, p.y)
    return res

def fancy_thresholding(im, debug=False):
    r, g, b = cv2.split(im)
    c = np.maximum(r, np.maximum(g, b))
    if debug:
        cv2.imwrite('t1.png', c)

    # c = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    c = cv2.GaussianBlur(c, (0, 0), 2)
    if debug:
        cv2.imwrite('t2.png', c)

    mid = cv2.GaussianBlur(c, (0, 0), 8)
    c = (3 * (c.astype(np.float32) - mid.astype(np.float32)) + 128).clip(0, 255).astype(np.uint8)
    _, c = cv2.threshold(c, 128, 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imwrite('t3.png', c)

    c = cv2.erode(c, circ_dilation(1))
    if debug:
        cv2.imwrite('t4.png', c)

    # Fill in small contours
    _, contour, _ = cv2.findContours(c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        if cv2.contourArea(cnt) < 20000:
            cv2.drawContours(c, [cnt], 0, 255, -1)

    if debug:
        cv2.imwrite('t5.png', c)

    c = cv2.erode(c, circ_dilation(1))
    if debug:
        cv2.imwrite('t6.png', c)

    c = 255 - c
    return c


def resize(im, desiredWidth):
    scale = desiredWidth / max(im.shape[0], im.shape[1])
    newSize = (round(im.shape[1] * scale), round(im.shape[0] * scale))
    im = cv2.resize(im, newSize)
    return im


def bgr2rgb(color):
    return (color[2], color[1], color[0])


def create_color_classifier(bgr_colors):
    assasin = min(bgr_colors, key=lambda v: sum(v))
    flat = bgr_colors[:]
    flat.remove(assasin)

    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0)

    def classify(col):
        if np.sum(np.abs(np.array(assasin) - col)) < 0.01:
            return "a"

        col -= means
        col /= stds
        _, g, r = col
        angle = math.atan2(g, r)
        if abs(angle) > 0.65 * math.pi:
            return "b"
        elif angle < 0:
            return "r"
        else:
            return "c"

    return classify


def find_grid(fname, debug=False):
    im = cv2.imread(fname)
    if im is None:
        raise Exception("Could not read file: " + fname)

    # im = resize(im, 512)
    # gray = some_other_thresholding(im)

    im = resize(im, 1024)
    gray = fancy_thresholding(im, debug)
    gray = resize(gray, 512)
    im = resize(im, 512)

    # show(gray)
    # contIm, contours, hierarchy = cv2.findContours(gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # show(contIm)

    height, width = gray.shape

    areas = dfs_segmentation(gray, 50, 512**2 // 20)

    points = [((ar[1] + ar[3]) / 2, (ar[0] + ar[2]) / 2) for ar in areas]
    if debug:
        print(len(points))

    grid = fit_grid(points)

    if grid is None:
        return None

    gridcolors = [[None] * SIZE for _ in range(SIZE)]
    minsum = INF
    blackind = None
    for i in range(SIZE):
        for j in range(SIZE):
            p = grid[i][j]
            dists = []
            if i > 0:
                dists.append(dist(grid[i-1][j], p))
            if j > 0:
                dists.append(dist(grid[i][j-1], p))
            if i < SIZE-1:
                dists.append(dist(grid[i+1][j], p))
            if j < SIZE-1:
                dists.append(dist(grid[i][j+1], p))
            rad = int(sum(dists) / len(dists) / 3)
            X = int(p[0])
            Y = int(p[1])

            # Extract a small region around the tile
            patch = im[max(Y-rad, 0):min(Y+rad+1, height),max(X-rad,0):min(X+rad+1,width)]

            # If the cell is outside the picture, detection must have gone wrong.
            if patch.size == 0:
                return None

            # Take the mean of the RGB values in the image patch
            meanb, meang, meanr = np.mean(patch, axis=(0,1))

            if debug:
                print(X, Y, meanb, meang, meanr, rad)
            gridcolors[i][j] = (meanb, meang, meanr)
            if meanb + meang + meanr < minsum:
                minsum = meanb + meang + meanr
                blackind = (i, j)

    color_classifier = create_color_classifier([x for sub in gridcolors for x in sub])

    mat = []
    for i in range(SIZE):
        mrow = ''
        for j in range(SIZE):
            col = color_classifier(gridcolors[i][j])
            mrow += col
        mat.append(mrow)

    if debug:
        for row in grid:
            for co in row:
                x = int(co[0]) - 5
                y = int(co[1]) - 5
                for i in range(11):
                    for j in range(11):
                        gray[max(min(y+i, height-1), 0)][max(min(x+j, width-1), 0)] = 128

        for i in range(SIZE):
            for j in range(SIZE):
                x,y = grid[i][j]
                x = int(x)
                y = int(y)
                color = color_classifier(gridcolors[i][j])
                # realColor = gridcolors[i][j]
                realColor = color2rgb(color)
                realColor = bgr2rgb(realColor)
                cv2.drawMarker(im, (x,y), (0,0,0), markerSize=5, thickness=9, markerType=cv2.MARKER_DIAMOND, line_type=cv2.LINE_AA)
                cv2.drawMarker(im, (x,y), realColor, markerSize=5, thickness=7, markerType=cv2.MARKER_DIAMOND, line_type=cv2.LINE_AA)

    return mat


def color2rgb(c):
    if c == "b":
        return (0, 0, 255)
    if c == "r":
        return (255, 0, 0)
    if c == "c":
        return (128, 128, 128)
    if c == "a":
        return (0, 255, 255)
    return (255, 255, 255)


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
    grid = find_grid(sys.argv[1], debug=True)
    if grid is None:
        print("<no grid>")
    else:
        for row in grid:
            print(''.join(colorize(c) for c in row))
