import cv2
import sys
import numpy as np
from PIL import Image
import tesserocr


def show(im):
    Image.fromarray(im).show()


def rect_area(rect):
    return rect[2] * rect[3]


im = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
# show(blur1)
edges = cv2.Canny(blur1, 40, 100)
# show(edges)
# blur = cv2.GaussianBlur(edges, (41, 41), 0)

dilation_size = 6
element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                    (2 * dilation_size + 1, 2 * dilation_size + 1),
                                    (dilation_size, dilation_size))
dilated = cv2.dilate(edges, element)
# show(dilated)


# retval, labels = cv2.connectedComponents(dilated)
# print(retval, labels)
contIm, contours, hierarchy = cv2.findContours(dilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# im2 = im.copy()
# cv2.drawContours(im2, contours, -1, (255, 255, 255), 2)
# show(im2)

contours = [c for c in contours if cv2.contourArea(c) > 1000 and rect_area(cv2.boundingRect(c)) < 100000]

blacklist = [False] * len(contours)
for h in hierarchy:
    if h.item(3) >= 0:
        blacklist[h.item(3)] = True

contours = [c for i, c in enumerate(contours) if not blacklist[i]]

# im3 = im.copy()
# cv2.drawContours(im3, contours, -1, (255, 255, 255), 2)
# show(im3)


class Card:
    def __init__(self, points, word):
        self.points = points
        self.word = word

    def __repr__(self):
        return self.word


foundWords = []
with tesserocr.PyTessBaseAPI() as tess:
    cnt = 0
    for c in contours:
        rect = cv2.minAreaRect(c)
        if rect[2] < -45:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] + 90)
        if rect[2] > 45:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)

        # Skip rectangles that are rotated too much (not likely to be words)
        if abs(rect[2]) > 20:
            continue

        r = cv2.boxPoints(rect)

        width = int(rect[1][0])
        height = int(rect[1][1])
        h = np.array([[0, height], [0, 0], [width, 0], [width, height]], np.float32)
        transform = cv2.getPerspectiveTransform(r, h)
        warp = cv2.warpPerspective(blur1, transform, (width, height))

        # warp = ((warp - np.min(warp)).astype(np.float) * (255.0 / (np.max(warp) - np.min(warp)))).astype(np.uint8)

        i = Image.fromarray(warp)
        tess.SetImage(i)
        result = tess.GetUTF8Text().replace(" ", "").strip().upper()

        points = [[int(x), int(y)] for x,y in r]
        foundWords.append(Card(points, result))

        cnt += 1
        # if cnt <= 10:
        #    i.show()


wordList = [line.strip() for line in open('wordlist.txt')]
actualWords = [word for word in foundWords if word.word in wordList]
print(actualWords)

# cimg = im.copy()
# cv2.drawContours(cimg, contours, -1, (0, 255, 0), 2)
# for w in foundWords:
#     rects = np.array([w.points])
#     color = (255, 255, 255) if w in actualWords else (255, 0, 0)
#     cimg = cv2.polylines(cimg, rects, True, color, 2)
# show(cimg)