#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
nohup python3 training.py --train_path / --valid_path / > train.txt & #Modificare i paths

#nohup python3 bot.py > bot.txt &  VERIFICARE BOT.py