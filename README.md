# codenames-board-recognizer

## Dependencies
python3 (3.6.1 working), tesseract (3.05.01 working), pillow (4.1.1 working), tesserocr (2.2.1 working), cffi, opencv (>= 3.1)

On macOS:
~~~
brew install opencv tesseract
pip3 install pillow tesserocr cffi
./build-ffi.sh
~~~

On Linux (apt):
~~~
sudo add-apt-repository ppa:lkoppel/opencv; sudo apt update # Or however you get opencv3
sudo apt install tesseract-ocr tesseract-ocr-eng libleptonica-dev python3-opencv
pip3 install pillow tesserocr cffi
./build-ffi.sh
~~~

## Usage
~~~
python3 boardRecognizer.py examples/board/example1.jpg
~~~
