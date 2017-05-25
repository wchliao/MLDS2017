#!/bin/bash
python download_files.py
python generate_thought_vectors.py --caption_file=$1
python generate_images.py