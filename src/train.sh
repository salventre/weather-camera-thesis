CUDA_VISIBLE_DEVICES=3 nohup python3 src/training.py --train_path ./data/dataset/train/ --epochs 1 > log_train.txt &
nohup python3 utils/bot.py > bot.txt &