nohup python train_rcnn.py --mode predict_video --model output &> ./logs/$(date "+%Y%m%d_%H%M")_predict_video.out &
tail -f ./logs/$(date "+%Y%m%d_%H%M")_predict_video.out