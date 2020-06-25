nohup python train_rcnn.py --mode predict_image --model output &> ./logs/$(date "+%Y%m%d_%H%M")_predict_image.out &
tail -f ./logs/$(date "+%Y%m%d_%H%M")_predict_image.out