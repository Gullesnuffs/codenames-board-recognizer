# codenames-board-recognizer

## Dependencies
python3, tesseract, pillow, tesserocr, cffi, opencv

On macOS:
~~~
brew install tesseract
pip3 install pillow tesserocr cffi
./build-ffi.sh
~~~

On Linux (apt):
~~~
sudo add-apt-repository ppa:orangain/opencv; sudo apt update # Or however you get opencv3
sudo apt install tesseract-ocr tesseract-ocr-eng libleptonica-dev python3-opencv
pip3 install pillow tesserocr cffi
./build-ffi.sh
~~~

## Usage
~~~
python3 boardRecognizer.py examples/board/example1.jpg
~~~
