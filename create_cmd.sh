#!/bin/bash

set -eu

mkdir -p bin

find . -name "*.cu" -type f | while read -r cuda_file; do
    echo "--------------------"
    base_name=$(basename "$cuda_file")
    output_name="${base_name%.cu}"
    echo "nvcc $cuda_file -o bin/$output_name"
    echo "cd bin"
    echo "$output_name"
done

find . -name "*.c" -type f | while read -r c_file; do
    echo "--------------------"
    base_name=$(basename "$c_file")
    output_name="${base_name%.c}"
    echo "gcc $c_file -o bin/$output_name"
    echo "cd bin"
    echo "./${output_name}.exe"
done
