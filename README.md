## Project for trainnig detectron2 rcnn model with custom data
- Customizing image datasets as COCO format
- Trainning rcnn model with custom dataset
---
###1. Prepare (for building detectron2)
- python >= 3.6
- pytorch >= 1.4
- torchvision
- gcc & g++ >= 5
 ```
conda create -n detectron2 python=3.7
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install opencv
```
###2. Build detectron2
- Refer to : https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
- server
    ```
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ``` 
- local
    ```python
     git clone https://github.com/facebookresearch/detectron2.git
     python -m pip install -e detectron2
    ```
- macOS
    ```python
     CC=clang CXX=clang++ python -m pip install -e .
    ```

###3. Customizing dataset
- Refer COCO format : https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md 
- Transform data format of 'custom csv file' to COCO format
- code: /datasets/prepare_custom.py
    
###4. Register dataset and setting label
- Register dataset function to detectron2.data.DatasetCatalog, and then you can train model with custom data.
- code: /train_rcnn.py - set_dataset()
    
###5. Edit config value
- Edit your custom config value such like that (default):
    - NUM_WORKERS : 2
    - IMS_PER_BATCH : 2
    - BASE_LR : 0.00025
    - MAX_ITER : 40000
    - BATCH_SIZE_PER_IMAGE : 256
    - MODEL_WEIGHTS : 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
- code: /train_rcnn.py - set_config_custom()

###6. Run trainning
- You can train custom model as running this batch file : ./ptrain.bat
- After trainnnig model, If you want only evaluate image or video, you can just run batch file :
    - image eval : ./ppredict_image.bat
    - video eval : ./ppredict_video.bat
    
###7. Confirm results
- After done, the result files in output folder(such as './output_20200101_0000') such like that:
     ```
     20200625_1445_train/
    events.out.tfevents.1593063914.petopia-01.11056.0
    events.out.tfevents.1593078578.petopia-01.11056.1
    events.out.tfevents.1593078581.petopia-01.11056.2
    last_checkpoint
    metrics.json
    model_0004999.pth
    model_0009999.pth
    model_0014999.pth
    model_0019999.pth
    model_final.pth
     ```
- In prediction_folder(such as './20200625_1445_train'), there is eval result and trainning config set:
    ```python
    COCO_eval/         --> eval result of images
    config_all.txt     --> trainning config set
    config_custom.txt  --> trainning config set
    eval_result.txt    --> eval result of images
    prediction_multi/  --> predict result of images
    test_result.mp4    --> predict result of video
    ```