from __future__ import print_function
import os, sys, time
from PIL import Image, ImageFilter, ImageEnhance
from subprocess import call, DEVNULL
from termcolor import colored


def resize(im, max_width, max_height):
  im = im.resize((max_width, max_height), Image.LANCZOS)
  return im


def find_edges(im, threshold):
  im = im.filter(ImageFilter.GaussianBlur)
  im_edges = im.filter(ImageFilter.FIND_EDGES)
  im_edges = im_edges.filter(ImageFilter.GaussianBlur(3))
  # for _ in range(3):
    # im_edges = im_edges.filter(ImageFilter.GaussianBlur(2))
  im_edges = im_edges.convert('L')
  im_edges = im_edges.point(lambda x: 0 if x > 1 else 255, '1')
  im_edges = im_edges.convert('RGB')
  im_edges = im_edges.filter(ImageFilter.GaussianBlur(6))
  # for _ in range(8):
    # im_edges = im_edges.filter(ImageFilter.GaussianBlur(2))
  im_edges = im_edges.convert('L')
  im_edges = im_edges.point(lambda x: 0 if x <= threshold else 255, '1')
  return im_edges


def dfs_segmentation(rgb_im, im_edges, minimumArea):
  # im_edges.show()
  # rgb_im.show()
  # Do DFS from each white pixel to find regions of white pixels
  width, height = im_edges.size
  visited = [[False] * height for _ in range(width)]

  DX = [1, 0, -1, 0]
  DY = [0, 1, 0, -1]
  data = im_edges.getdata(0)
  rgb_data = rgb_im.getdata()
  for x in range(width):
    for y in range(height):
      if visited[x][y]:
        continue
      if data[y * width + x] == 0:
        continue
      totR = 0
      totG = 0
      totB = 0
      totPixels = 0
      q = [(x, y)]
      minX = x
      maxX = x
      minY = y
      maxY = y
      visited[x][y] = True
      while len(q) > 0:
        cx, cy = q.pop()
        for i in range(4):
          nx = cx + DX[i]
          ny = cy + DY[i]
          if nx < 0 or ny < 0 or nx >= width or ny >= height:
            continue
          if data[ny * width + nx] == 0:
            continue
          if visited[nx][ny]:
            continue
          r, g, b = rgb_data[ny * width + nx]
          totR += r
          totG += g
          totB += b
          totPixels += 1
          visited[nx][ny] = True
          minX = min(minX, nx)
          maxX = max(maxX, nx)
          minY = min(minY, ny)
          maxY = max(maxY, ny)
          q.append((nx, ny))
      regionWidth = maxX - minX + 1
      regionHeight = maxY - minY + 1
      area = regionWidth * regionHeight
      if area < minimumArea:
        continue
      box = (minX, minY, maxX, maxY)
      # print("found box " + str(box) + " " + str(area))
      box = clamp_aabb(expand_aabb(box, 3), width, height)
      averageR = totR/totPixels
      averageG = totG/totPixels
      averageB = totB/totPixels
      yield (averageR, averageG, averageB, box)


def tocard(box):
  r = box[0]
  g = box[1]
  b = box[2]
  if r > g*2 and r > b*2 and r > 70:
    result = "r"
  elif b > r*2 and b > g*0.8 and b > 60:
    result = "b"
  elif r > 40 and g > 40 and b > 40:
    result = "c"
  else:
    result = "a"
  box = box[3]
  midX = (box[0] + box[2])/2
  midY = (box[1] + box[3])/2
  return Card(result, (midX, midY))


def bounding_box_area(box):
  return (box[3][2] - box[3][0]) * (box[3][3] - box[3][1])


def scale_aabb(box, scale):
  """ Scales the size of the bounding box around its center """
  xmin, ymin, xmax, ymax = box
  dx = (xmax - xmin) * (scale - 1) * 0.5
  dy = (ymax - ymin) * (scale - 1) * 0.5
  return (xmin - dx, ymin - dy, xmax + dx, ymax + dy)


def expand_aabb(box, expansion):
  """ Expands the bounding box by 'expansion' units in all directions """
  xmin, ymin, xmax, ymax = box
  dx = expansion
  dy = expansion
  return (xmin - dx, ymin - dy, xmax + dx, ymax + dy)


def clamp_aabb(box, width, height):
  return (
      max(min(box[0], width), 0),
      max(min(box[1], height), 0),
      max(min(box[2], width), 0),
      max(min(box[3], height), 0)
  )


def aabb_overlap(box1, box2):
  xmin1, ymin1, xmax1, ymax1 = box1
  xmin2, ymin2, xmax2, ymax2 = box2
  return not (xmin1 > xmax2 or ymin1 > ymax2 or xmax1 < xmin2 or ymax1 < ymin2)


def filter_outer_boxes(boxes):
  """ Removes bounding boxes that significantly overlap other smaller bounding boxes """
  boxes.sort(key=lambda box: bounding_box_area(box))
  result = []
  for box in boxes:
    if all(not(aabb_overlap(scale_aabb(box[3], 0.8), scale_aabb(b[3], 0.8))) for b in result):
      # print(str(box) + " - YES " + tocard(box).word)
      result.append(box)
    # else:
      # print(str(box) + " - NO")
  return result


class Card:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos

    def __repr__(self):
        return self.word


def fit_grid_to_words(cards, width, height):
  def get_grid_score(x0, y0, dx, dy):
    wordIndex = [[-1] * 5 for _ in range(5)]
    distances = []
    used = [False] * len(cards)
    for ind in range(len(cards)):
      x, y = cards[ind].pos
      for i in range(5):
        for j in range(5):
          sx = x0 + dx * i
          sy = y0 + dy * j
          Dx = x - sx
          Dy = y - sy
          dis = Dx*Dx + Dy*Dy
          distances.append((dis, ind, i, j))
    distances = sorted(distances)
    score = 0
    maxDis = 40000
    numCivilian = 0
    numBlue = 0
    numRed = 0
    numAssassin = 0
    for (dis, ind, i, j) in distances:
      if dis > maxDis:
        break
      if used[ind]:
        continue
      if wordIndex[i][j] != -1:
        continue
      if cards[ind].word == 'c':
        if numCivilian == 7:
          continue
        numCivilian += 1
      if cards[ind].word == 'b':
        if numBlue == 9 or (numRed == 9 and numBlue == 8):
          continue
        numBlue += 1
      if cards[ind].word == 'r':
        if numRed == 9 or (numBlue == 9 and numRed == 8):
          continue
        numRed += 1
      if cards[ind].word == 'a':
        if numAssassin == 1:
          continue
        numAssassin += 1
      score += dis
      wordIndex[i][j] = ind
      used[ind] = True
    for i in range(5):
      for j in range(5):
        if wordIndex[i][j] == -1:
          score += maxDis
    return (score, wordIndex)

  def hillClimb():
    jump = width / 2
    x0 = width / 10
    y0 = height / 10
    dx = width / 10
    dy = height / 10
    bestScore = 1e30
    while jump > width/200:
      jump /= 2
      for iterations in range(2):
        for newx0 in [x0 - jump, x0, x0 + jump]:
          for newdx in [dx - jump, dx, dx + jump]:
            newdx = max(0, newdx)
            (score, wordIndex) = get_grid_score(newx0, y0, newdx, dy)
            if score < bestScore:
              bestScore = score
              bestWordIndex = wordIndex
              x0 = newx0
              dx = newdx
        for newy0 in [y0 - jump, y0, y0 + jump]:
          for newdy in [dy - jump, dy, dy + jump]:
            newdy = max(0, newdy)
            (score, wordIndex) = get_grid_score(newx0, y0, newdx, dy)
            (score, wordIndex) = get_grid_score(x0, newy0, dx, newdy)
            if score < bestScore:
              bestScore = score
              bestWordIndex = wordIndex
              y0 = newy0
              dy = newdy
    return bestWordIndex

  bestWordIndex = hillClimb()
  grid = []
  for i in range(5):
    grid.append([])
    for j in range(5):
      if bestWordIndex[j][i] == -1:
        grid[i].append("")
      else:
        grid[i].append(cards[bestWordIndex[j][i]].word)

  return grid


def find_words(imagePath):
  im = Image.open(imagePath)
  im = resize(im, 2500, 1500)
  width, height = im.size

  rgb_im = im.convert("RGB")

  im_edges = find_edges(im, 200)

  im_edges = resize(im_edges, 1250, 750)
  rgb_im = resize(rgb_im, 1250, 750)
  width, height = im_edges.size
  totalArea = width * height

  minimumPartOfImage = 0.0001
  minimumArea = totalArea * minimumPartOfImage
  boxes = list(dfs_segmentation(rgb_im, im_edges, minimumArea))
  boxes = filter_outer_boxes(boxes)
  im_edges.show()
  print(len(boxes))
  if len(boxes) > 35:
    im_edges = find_edges(im, 140)
    im_edges = resize(im_edges, 1250, 750)
    boxes = list(dfs_segmentation(rgb_im, im_edges, minimumArea))
    boxes = filter_outer_boxes(boxes)
    print(len(boxes))
    im_edges.show()
    if len(boxes) < 24:
      im_edges = find_edges(im, 170)
      im_edges = resize(im_edges, 1250, 750)
      boxes = list(dfs_segmentation(rgb_im, im_edges, minimumArea))
      boxes = filter_outer_boxes(boxes)

  # rgb_im = ImageEnhance.Contrast(rgb_im).enhance(1.4)
  # for box in boxes:
    # rgb_im.crop(box[3]).show()

  foundWords = [tocard(box) for box in boxes]
  foundWords = [w for w in foundWords if w.word]

  grid = fit_grid_to_words(foundWords, width, height)

  words = [w.word for w in foundWords]
  return words, grid


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("usage: python3 secretRecognizer.py image")
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
