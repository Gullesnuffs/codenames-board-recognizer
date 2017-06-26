import cv2
import numpy as np
from PIL import Image
import sys
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def show(im):
  Image.fromarray(im).show()


def resize(im, desiredWidth):
  scale = desiredWidth / max(im.shape[0], im.shape[1])
  newSize = (round(im.shape[1] * scale), round(im.shape[0] * scale))
  im = cv2.resize(im, newSize)
  return im


img2 = cv2.imread(sys.argv[1])
img2 = resize(img2, 1024)
# img2 = cv2.GaussianBlur(img2, (0, 0), 8)

path = "examples/templates/secret4.jpg"
template = cv2.imread(path)
template = resize(template, 1024)

def preprocess(im):
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

  dilation = 3
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
  dilation = 4
  element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilation + 1, 2 * dilation + 1),
                                      (dilation, dilation))
  c = cv2.erode(c, element)
  cv2.imwrite('t6.png', c)

  return c


template = preprocess(template)
c = preprocess(img2)
# show(c)
# show(template)

c = cv2.blur(c, (20, 20))
template = cv2.blur(template, (30, 30))

# show(c)
# show(template)


# # edges = cv2.Canny(mid, 30, 60)
# # show(edges)
# exit(0)
#
# col = [255, 0, 0]
#
# mid = cv2.GaussianBlur(c, (0,0), 50)
# c = (3 * (c.astype(np.float32) - mid.astype(np.float32)) + 128).clip(0, 255).astype(np.uint8)
#
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# mid = cv2.GaussianBlur(template, (0,0), 50)
# template = (3 * (template.astype(np.float32) - mid.astype(np.float32)) + 128).clip(0, 255).astype(np.uint8)
#
# c = cv2.GaussianBlur(c, (0,0), 8)
# template = cv2.GaussianBlur(template, (0,0), 8)
# show(c)
# show(template)

bestLocs = []
mins = []
# sizes = [50, 60, 75, 85, 90, 93, 94, 95, 97, 100, 110, 130, 150, 160, 180, 195, 210, 240, 260, 300, 350, 400, 430, 450, 470, 500, 600, 700]
sizes = range(150, 1024, 20)
sizes = [s for s in sizes if s < c.shape[0] and s < c.shape[1]]
print(sizes)
# sizes = [430, 450, 470]
for sz in sizes:
  tc = resize(template, sz)

  w = tc.shape[1]
  h = tc.shape[0]

  # show(r)
  # show(tr)

  res = cv2.matchTemplate(c, tc, cv2.TM_SQDIFF)

  res = res * 0.001
  res /= (w -1) * (h - 1)
  # show(res)
  # res = (res - np.min(res))
  # res = (res * (255 / (np.max(res) - np.min(res)))).astype(np.uint8)

  threshold = 80

  dilation = 5
  # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
  #                                     (2 * dilation + 1, 2 * dilation + 1),
  #                                     (dilation, dilation))
  # dilated = cv2.dilate(res, element)
  # res *= np.clip(cv2.compare(res, dilated, cv2.CMP_GE), 0, 1)

  # v, p = np.max(res)
  v, _, p, _ = cv2.minMaxLoc(res)
  bestLocs.append((v, p, h, w))
  mins.append(v)

  # cv2.rectangle(img2, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), col, 2)

  # list(loc).sort(key=lambda p: res.item(p[0], p[1]))

  # res *= loc

  # for pt in zip(*loc[::-1]):
  #   value = res.item(pt[1], pt[0]) / (sz*sz)
  #   bestLocs.append((value, pt, w, h))
  #   break

print(mins)

if True:
  bestLocs.sort()

  mn = min(mins)
  for v, pt, w, h in bestLocs[:1]:
    print(v)
    brightness = round((mn / v)**2 * 255)
    col = (brightness, brightness, brightness)
    if v == mn:
      col = (0, 0, 255)
    cv2.rectangle(img2, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), col, 2)

  # show(img2)
  cv2.imwrite('res.png', img2)

if False:
  rcParams['font.family'] = 'serif'
  rcParams['font.serif'] = ['CMU Serif']
  rcParams['font.size'] = 24
  rcParams['mathtext.default'] = 'regular'

  fig, ax = plt.subplots(figsize=(12, 12))
  ax.scatter(x=sizes, y=list((11 - np.array(mins)) + 11) , s=300)
  pdf = PdfPages('out.pdf')
  pdf.savefig(fig)
  pdf.close()
