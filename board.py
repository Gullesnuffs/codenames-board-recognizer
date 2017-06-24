import cv2
import sys
import numpy as np
import scipy.optimize
from PIL import Image
import tesserocr
import scipy.optimize


def show(im):
    Image.fromarray(im).show()


def rect_area(rect):
    return rect[2] * rect[3]


class Card:
    def __init__(self, word, rect):
        self.word = word
        self.rect = rect
        self.pos = rect[0]
        self.rect_vertices = [[int(x), int(y)] for x, y in cv2.boxPoints(self.rect)]

    def __repr__(self):
        return self.word


def calculate_naive_bounding_rect(cards):
    topleft = min(cards, key=lambda c: c.pos[0] + c.pos[1])
    topright = min(cards, key=lambda c: -c.pos[0] + c.pos[1])
    botleft = min(cards, key=lambda c: c.pos[0] - c.pos[1])
    botright = min(cards, key=lambda c: -c.pos[0] - c.pos[1])
    return [botleft.pos, botright.pos, topright.pos, topleft.pos]


def poly_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_optimized_bounding_rect(words, im):
    width = im.shape[1]
    height = im.shape[0]

    positions = [w.pos for w in words]
    padding = 40

    def loss(x):
        cost = 0
        x = np.reshape(x, [4, 2]).astype(np.float32)
        for p in positions:
            d = -cv2.pointPolygonTest(x, p, True)
            d += padding
            d = max(d, 0)
            cost += d + d*d*2.5

        cost += poly_area(x) * 0.01
        return cost

    x0 = np.array([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0],
    ]).reshape([8])

    xf = scipy.optimize.fmin_cg(loss, x0, epsilon=0.001)
    return xf.reshape([4, 2]).tolist()


def fit_words_to_grid(cards, bounding_rect):
    output = [[""] * 5 for _ in range(5)]
    if not cards:
        return output

    def c(card):
        return complex(card.pos[0], card.pos[1])

    positions = list(map(c, cards))
    costs = [[] for _ in range(len(positions))]
    # topleft, topright, botright, botleft = [complex(v[0], v[1]) for v in bounding_rect]
    botleft, botright, topright, topleft = [complex(v[0], v[1]) for v in bounding_rect]
    for i in range(5):
        for j in range(5):
            a = topleft + (j / 4.0) * (topright - topleft)
            b = botleft + (j / 4.0) * (botright - botleft)
            p = b * (i / 4.0) + a * (1 - i / 4.0)
            for k in range(len(positions)):
                dis = (p - positions[k])
                costs[k].append(dis.imag * dis.imag + dis.real * dis.real)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    for i, j in zip(row_ind, col_ind):
        output[j//5][j%5] = cards[i].word
    return output


def find_contours(im, dilation, min_contour_area, max_contour_area):
    edges = cv2.Canny(im, 40, 100)
    # show(edges)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * dilation + 1, 2 * dilation + 1),
                                        (dilation, dilation))
    dilated = cv2.dilate(edges, element)
    # show(dilated)

    contIm, contours, hierarchy = cv2.findContours(dilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # im2 = im.copy()
    # cv2.drawContours(im2, contours, -1, (255, 255, 255), 2)
    # show(im2)

    def valid_contour(contour):
        area = rect_area(cv2.boundingRect(contour))
        return area > min_contour_area and area < max_contour_area

    contours = [c for c in contours if valid_contour(c)]

    blacklist = []
    for h in hierarchy:
        if h.item(3) >= 0:
            print(h.item(3))
            blacklist.append(h.item(3))

    contours = [c for i, c in enumerate(contours) if i not in blacklist]
    contours.sort(key=lambda c: c[0].item(1))
    return contours


def find_text_in_contours(contours, image, word_list):
    def align_rect_horizontal(rect):
        if rect[2] < -45:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] + 90)
        if rect[2] > 45:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
        return rect

    def extract_rect_with_perspective(rect, scale):
        rectVertices = cv2.boxPoints(rect)
        width = int(rect[1][0] * scale)
        height = int(rect[1][1] * scale)
        h = np.array([[0, height], [0, 0], [width, 0], [width, height]], np.float32)
        transform = cv2.getPerspectiveTransform(rectVertices, h)
        warp = cv2.warpPerspective(image, transform, (width, height))
        return warp

    noSpaces2words = dict((w.upper().replace(" ", ""), w) for w in word_list)
    with tesserocr.PyTessBaseAPI() as tess:
        for c in contours:
            rect = align_rect_horizontal(cv2.minAreaRect(c))

            # Skip rectangles that are rotated too much (not likely to be words)
            if abs(rect[2]) > 20:
                continue

            # Scale up the patch by 2x. This improves OCR accuracy
            scale = 2
            patch = extract_rect_with_perspective(rect, scale)
            patch = ((patch - np.min(patch)).astype(np.float) * (255.0 / (np.max(patch) - np.min(patch)))).astype(np.uint8)

            patch = cv2.resize(patch, (patch.shape[1]*2, patch.shape[0]*2))
            # show(patch)
            tess.SetImage(Image.fromarray(patch))

            result = tess.GetUTF8Text().replace(" ", "").strip().upper()

            if result in noSpaces2words:
                result = noSpaces2words[result]

            yield Card(result, rect)


def unique(words):
    uniqueWords = []
    for word in words:
        if all(w.word != word.word for w in uniqueWords):
            uniqueWords.append(word)
    return uniqueWords


def draw_match_grid(words, rect, im):
    im = im.copy()

    # cv2.drawContours(cimg, contours, -1, (255, 255, 255), 2)
    for w in words:
        rects = np.array([w.rect_vertices])
        color = (77, 175, 74)
        cv2.polylines(im, rects, True, color, 2)

    rect = np.array(rect).reshape([1,4,2]).astype(np.int64)
    cv2.polylines(im, rect, True, (255, 255, 255), 2)
    show(im)


def find_words(imagePath):
    im = cv2.imread(imagePath)
    desiredWidth = 2048
    scale = desiredWidth / max(im.shape[0], im.shape[1])
    newSize = (round(im.shape[1] * scale), round(im.shape[0] * scale))
    im = cv2.resize(im, newSize)

    # Make the settings independent of size
    length_unit = desiredWidth / 2048
    area_unit = length_unit * length_unit

    min_contour_area = 1000
    max_contour_area = 100000
    dilation = 8
    blur_amount = 3

    blur = round(blur_amount * length_unit)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray, (blur, blur), 0)
    blur2 = cv2.GaussianBlur(gray, (1, 1), 0)

    # blur1 = (np.clip((blur1.copy().astype(np.float) - 100)*(255 / (255 - 100))*(255/150), 0, 255)).astype(np.uint8)
    # show(blur1)

    contours = find_contours(blur1,
                             dilation=round(dilation * length_unit),
                             min_contour_area=min_contour_area * area_unit,
                             max_contour_area=max_contour_area * area_unit,
                             )

    # im3 = im.copy()
    # cv2.drawContours(im3, contours, -1, (255, 255, 255), 2)
    # show(im3)

    word_list = [line.strip() for line in open('wordlist.txt')]
    foundWords = list(find_text_in_contours(contours, blur2, word_list))
    uniqueWords = unique(foundWords)
    actualWords = [word for word in uniqueWords if word.word in word_list]
    print(len(actualWords))

    bounding_rect = calculate_optimized_bounding_rect(actualWords, im)
    draw_match_grid(actualWords, bounding_rect, im)
    grid = fit_words_to_grid(actualWords, bounding_rect)
    for row in grid:
        print(row)

    # cimg = im.copy()
    # cv2.drawContours(cimg, contours, -1, (255, 255, 255), 2)
    # for w in foundWords:
    #     rects = np.array([w.rect_vertices])
    #     color = (77,175,74) if w in actualWords else (228,26,28)
    #     cv2.polylines(cimg, rects, True, color, 2)
    # show(cimg)
    # rect = opt(actualWords, im)
    
    return [w.word for w in actualWords], grid


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 boardRecognizer.py image")
        exit(0)

    find_words(sys.argv[1])
