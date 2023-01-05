#!/bin/bash

#cmake -B build -S . -DCMAKE_CXX_COMPILER=clang -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Debug
cmake -B build -S . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug
