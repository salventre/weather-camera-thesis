#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python3 src/training.py --train_path ./data/dataset/train/ --epochs 1
#nohup python3 src/bot.py > bot.txt &