#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
nohup python3 src/generate_results.py --dataset src/partition_info/test_filenames.txt --output_path ./data/doc/results.txt --model_path ./data/checkpoint/model-epoch_05.hdf5 > log_test.txt