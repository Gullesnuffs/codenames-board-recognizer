import os
import boardRecognizer

path = "examples"
samples = []  # type: List[float]
for file in os.listdir(path):
    words, grid = boardRecognizer.find_words(path + "/" + file)
    score = len(words)
    print(score)
    samples.append(score)

finalScore = sum(samples)/len(samples)
print("Final score: " + str(finalScore))
