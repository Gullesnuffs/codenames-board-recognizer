#!/bin/sh
g++ -c -fPIC -fno-exceptions -fno-rtti -O2 -Wall -std=c++11 grid.cpp -o grid.o
python3 grid_build.py
