#!/bin/bash
python src/download_files.py
python src/generate_thought_vectors.py --caption_file=$1
python src/generate_images.py