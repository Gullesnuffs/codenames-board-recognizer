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
sudo apt install tesseract-ocr tesseract-ocr-eng
pip3 install pillow tesserocr
~~~

## Usage
~~~
python3 boardRecognizer.py examples/example_board1.jpg
~~~