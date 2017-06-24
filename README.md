# codenames-board-recognizer

## Dependencies
python3, tesseract, pillow, tesserocr

On macOS:
~~~
brew install tesseract
pip3 install pillow tesserocr
~~~

On Linux (apt):
~~~
sudo apt install tesseract-ocr tesseract-ocr-eng libleptonica-dev
pip3 install pillow tesserocr
~~~

## Usage
~~~
python3 boardRecognizer.py examples/board/example1.jpg
~~~
