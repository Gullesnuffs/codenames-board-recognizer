from __future__ import print_function
import os, sys, time
from PIL import Image, ImageFilter, ImageEnhance
from subprocess import call

if len(sys.argv) != 2:
  print("usage: python3 boardRecognizer.py image")
  exit(0)

wordListFile = open('wordlist.txt', 'r')
wordList = []
for line in wordListFile:
  wordList.append(line.strip())

im = Image.open(sys.argv[1])
sz = im.size
width = sz[0]
height = sz[1]
coordinateFactor = 1
if width > 3000:
  im = im.resize((width//2, height//2))
  width //= 2
  height //= 2
  coordinateFactor = 1
rgb_im = im.convert("RGB")
totalArea = width * height
im_edges = im.filter(ImageFilter.FIND_EDGES)
im_edges = im_edges.filter(ImageFilter.GaussianBlur)
im_edges = im_edges.filter(ImageFilter.GaussianBlur)
im_edges = im_edges.filter(ImageFilter.GaussianBlur)
im_edges.show()
for x in range(width):
  for y in range(height):
    r, g, b = im_edges.getpixel((x, y))
    brightness = r+g+b
    if brightness > 30:
      im_edges.putpixel((x, y), (255, 255, 255))
    else:
      im_edges.putpixel((x, y), (0, 0, 0))
im_edges.show()

# Do BFS from each white pixel to find regions of white pixels
visited = []  # type: List[List[bool]]
for x in range(width):
  visited.append([])
  for y in range(height):
    visited[x].append(False)

foundWords = []  # type: List[str]
foundWordsWithPositions = []
for x in range(width):
  for y in range(height):
    if visited[x][y]:
      continue
    if im_edges.getpixel((x, y))[0] == 0:
      continue
    q = [(x, y)]
    minX = x
    maxX = x
    minY = y
    maxY = y
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    visited[x][y] = True
    while len(q) > 0:
      cx = q[-1][0]
      cy = q[-1][1]
      q.pop()
      for i in range(4):
        nx = cx + dx[i]
        ny = cy + dy[i]
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
          continue
        if im_edges.getpixel((nx, ny))[0] == 0:
          continue
        if visited[nx][ny]:
          continue
        visited[nx][ny] = True
        minX = min(minX, nx)
        maxX = max(maxX, nx)
        minY = min(minY, ny)
        maxY = max(maxY, ny)
        q.append((nx, ny))
    regionWidth = maxX - minX
    regionHeight = maxY - minY
    area = (regionWidth+1)*(regionHeight+1)
    minimumPartOfImage = 0.0005
    minimumArea = totalArea * minimumPartOfImage
    if area < minimumArea:
      continue
    print(str(minX) + ', ' + str(minY) + ' ' + str(maxX) + ', ' + str(maxY))
    minX = max(minX-10, 0)
    minY = max(minY-10, 0)
    maxX = min(maxX+10, width)
    maxY = min(maxY+10, height)
    box = (minX * coordinateFactor, minY * coordinateFactor, maxX * coordinateFactor, maxY * coordinateFactor)
    region = rgb_im.crop(box)
    enhancer = ImageEnhance.Contrast(region)
    region = enhancer.enhance(1.4)
    #region.show()
    region.save("cropped_file.png")
    call(["tesseract", "cropped_file.png", "output"])
    resultFile = open("output.txt", 'r')
    result = resultFile.read()  # type: str
    resultFile.close()
    os.remove("output.txt")
    os.remove("cropped_file.png")
    result = result.strip().upper()
    if result not in foundWords:
      if result in wordList:
        midX = (minX+maxX)/2
        midY = (minY+maxY)/2
        foundWords.append(result)
        foundWordsWithPositions.append((midY*10+midX, result))
      else:
        print(result)
foundWordsWithPositions = sorted(foundWordsWithPositions)
foundWords = []
for pair in foundWordsWithPositions:
  foundWords.append(pair[1])
print("Found " + str(len(foundWords)) + " words!")
print(foundWords)
