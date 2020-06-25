nohup python train_rcnn.py --mode train &> ./logs/$(date "+%Y%m%d_%H%M")_train.out &
tail -f ./logs/$(date "+%Y%m%d_%H%M")_train.out
