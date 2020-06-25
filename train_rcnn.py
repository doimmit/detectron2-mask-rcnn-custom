import os
import cv2
import tqdm
from datetime import datetime
import argparse

from datasets.prepare_custom import get_custom_dataset, get_target_labels
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.checkpoint import DetectionCheckpointer

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='', help='ex) train')
parser.add_argument('--model', type=str, default='', help='ex) output_20200624_1655')

target = 'multi'


def set_dataset():
    for d in ["train", "test"]:
        DatasetCatalog.register("custom/" + d, lambda d=d: get_custom_dataset(target, "custom/" + d))
        MetadataCatalog.get("custom/" + d).set(thing_classes=get_target_labels()[target])


def set_config_custom(output_folder, mode, now_date):
    logger.debug(f" Set configs")
    config_custom = {
        'DATASET': '20200625_103328_8posture',
        'AUG_IN_DATASET': 'X',
        'CLASS_LIST': str(get_target_labels()[target]),
        'NUM_WORKERS': 2,
        # 2
        # Number of data loading threads

        'IMS_PER_BATCH': 4,
        # 2
        # Number of images per batch across all machines.
        # If we have 16 GPUs and IMS_PER_BATCH = 32,
        # each GPU will see 2 images per batch.

        'BASE_LR': 0.00025,
        # 0.00025

        'MAX_ITER': 20000,
        # 40000

        'BATCH_SIZE_PER_IMAGE': 512,
        # 256
        # RoI minibatch size *per image* (number of regions of interest [ROIs])
        # Total number of RoIs per training minibatch =
        #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
        # E.g., a common configuration is: 512 * 16 = 8192
        'MODEL_WEIGHTS': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'TEST_EVAL_PERIOD': 100
    }
    os.makedirs(os.path.join(output_folder, f'{now_date}_{mode}'), exist_ok=True)
    seq_path = os.path.join(output_folder, f'{now_date}_{mode}', f'config_custom.txt')
    wf = open(seq_path, 'w', newline='', encoding='utf-8')
    for k, v in config_custom.items():
        wf.write(f'{str(k)}: {str(v)}\n')
    wf.write(f'\n')
    wf.write(f'TARGET_OBJECT: {str(target)}\n')
    wf.write(f'START_TIME: {str(datetime.now())}\n')
    wf.close()

    return config_custom


def set_config_all(cfg, config_custom, output_folder, mode, now_date):
    ## 1. https://rosenfelder.ai/Instance_Image_Segmentation_for_Window_and_Building_Detection_with_detectron2/
    cfg.merge_from_file(f'./configs/{config_custom["MODEL_WEIGHTS"]}')
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_custom["MODEL_WEIGHTS"])

    ## 2. https://towardsdatascience.com/detectron2-the-basic-end-to-end-tutorial-5ac90e2f90e3
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

    cfg.DATASETS.TRAIN = ("custom/train",)
    cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = config_custom['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = (
        config_custom['IMS_PER_BATCH']
    )
    cfg.SOLVER.BASE_LR = (
        config_custom['BASE_LR']
    )
    # cfg.SOLVER.WARMUP_ITERS = 1000 # the learning rate starts from 0 and goes to the preset one for this number of iterations
    cfg.SOLVER.MAX_ITER = (
        config_custom['MAX_ITER']
    )
    # cfg.SOLVER.STEPS = (1000, 1500) # the checkpoints (number of iterations) at which the learning rate will be reduced by GAMMA
    # cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        config_custom['BATCH_SIZE_PER_IMAGE']
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(get_target_labels()[target])  # 4 action class (none,eating,peeing,pooping)
    # cfg.MODEL.DEVICE = "cuda:0,1"
    cfg.TEST.EVAL_PERIOD = config_custom['TEST_EVAL_PERIOD']

    cfg.OUTPUT_DIR=output_folder
    ## write config list
    seq_path = os.path.join(output_folder, f'{now_date}_{mode}', f'config_all.txt')
    wf = open(seq_path, 'w', newline='', encoding='utf-8')
    wf.write(f'{str(cfg)}')
    wf.close()

    return cfg


def train_model(cfg, output_folder, mode, now_date):
    logger.debug(f" Start train model")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # trainer.register_hooks(
    #     [hooks.EvalHook(0, lambda: test_with_TTA(cfg, trainer.model))]
    # )
    # 실시간 로스율 계산하기 참고: https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e

    seq_path = os.path.join(output_folder, f"{now_date}_{mode}", f'config_custom.txt')
    wf = open(seq_path, 'a', newline='', encoding='utf-8')
    wf.write(f'END_TIME_TRAIN: {str(datetime.now())}\n')
    logger.debug(f" End to train:{str(datetime.now())}")
    wf.close()

    return True


def make_prediction_image(cfg, output_folder, mode, now_date):
    logger.debug(f" Start evaluating image")
    os.makedirs(os.path.join(output_folder, f"{now_date}_{mode}", f"prediction_{target}"), exist_ok=True)

    cfg.MODEL.WEIGHTS = os.path.join(output_folder, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("custom/test",)

    # Evaluate test datasets
    model = DefaultTrainer(cfg).build_model(cfg)
    DetectionCheckpointer(model, save_dir=output_folder).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    evaluator = COCOEvaluator("custom/test", cfg, False,
                              output_dir=os.path.join(output_folder, f"{now_date}_{mode}", 'COCO_eval'))
    val_loader = build_detection_test_loader(cfg, "custom/test")
    order_dicts = inference_on_dataset(DefaultTrainer(cfg).model, val_loader, evaluator)

    seq_path = os.path.join(output_folder, f"{now_date}_{mode}", f'eval_result.txt')
    wf = open(seq_path, 'w', newline='', encoding='utf-8')
    wf.write(f'{str(order_dicts)}:\n')
    wf.close()

    # Predict test dataset
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_custom_dataset(target, "custom/test")
    for i, d in enumerate(dataset_dicts):
        logger.debug(f"predicting...{d['file_name']}")
        im = cv2.imread(d["file_name"])
        file_name = os.path.basename(d["file_name"])
        outputs = predictor(im)

        ## 1 https://rosenfelder.ai/Instance_Image_Segmentation_for_Window_and_Building_Detection_with_detectron2/
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("custom/test"),
            scale=1,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(output_folder, f"{now_date}_{mode}", f"prediction_{target}", file_name),
                    v.get_image())

    logger.debug(f" End evaluating image")

    seq_path = os.path.join(output_folder, f"{now_date}_{mode}", f'config_custom.txt')
    wf = open(seq_path, 'a', newline='', encoding='utf-8')
    wf.write(f'END_TIME_EVAL_IMAGE: {str(datetime.now())}\n')
    logger.debug(f" End to train:{str(datetime.now())}")
    wf.close()

    return True


# https://stackoverflow.com/questions/60663073/how-can-i-properly-run-detectron2-on-videos
def make_prediction_video(cfg, output_folder, mode, now_date):
    logger.debug(f" Start evaluating video")

    # Extract video properties
    video = cv2.VideoCapture('./test_video.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_writer = cv2.VideoWriter(os.path.join(output_folder, f"{now_date}_{mode}", f'test_result.mp4'),
                                   fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second),
                                   frameSize=(width, height), isColor=True)

    # Initialize predictor
    cfg.MODEL.WEIGHTS = os.path.join(output_folder, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("custom/test",)
    predictor = DefaultPredictor(cfg)

    # Initialize visualizer
    v = VideoVisualizer(MetadataCatalog.get("custom/test"), ColorMode.IMAGE)

    # Create a cut-off for debugging
    # num_frames = 5000

    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(run_on_video(video, v, num_frames, predictor), total=num_frames):
        # Write test image
        # cv2.imwrite(os.path.join(output_folder, 'test_result.png'), visualization)

        # Write to video file
        video_writer.write(visualization)

    # Release resources
    video.release()
    video_writer.release()
    # cv2.destroyAllWindows()
    logger.debug(f" End evaluating video")

    seq_path = os.path.join(output_folder, f"{now_date}_{mode}", f'config_custom.txt')
    wf = open(seq_path, 'a', newline='', encoding='utf-8')
    wf.write(f'END_TIME_EVAL_VIDEO: {str(datetime.now())}\n')
    logger.debug(f" End to train:{str(datetime.now())}")
    wf.close()

    return True


def run_on_video(video, v, maxFrames, predictor):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """
    readFrames = 0
    while True:
        hasFrame, frame = video.read()

        if not hasFrame:
            break

        # Get prediction results for this frame
        outputs = predictor(frame)
        # Make sure the frame is colored
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Draw a visualization of the predictions using the video visualizer
        visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        # Convert Matplotlib RGB format to OpenCV BGR format
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        yield visualization
        readFrames += 1
        if readFrames > maxFrames:
            break


def init(mode, model=None):
    cfg_origin = get_cfg()
    today_date = datetime.now().strftime("%Y%m%d_%H%M")
    if model is not None and model != '':
        output_folder = os.path.splitext(cfg_origin.OUTPUT_DIR)[0].replace('output', model)
    else:
        output_folder = os.path.join(f'{cfg_origin.OUTPUT_DIR}_{today_date}_{target}')
        os.makedirs(output_folder, exist_ok=True)

    config_custom = set_config_custom(output_folder, mode, now_date)
    cfg = set_config_all(cfg_origin, config_custom, output_folder, mode, now_date)

    return cfg, output_folder


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    args = parser.parse_args()
    mode = args.mode
    model = args.model
    now_date = datetime.now().strftime("%Y%m%d_%H%M")
    cfg, output_folder = init(mode, model)

    set_dataset()
    if mode == 'train':
        train_model(cfg, output_folder, mode, now_date)
        make_prediction_image(cfg, output_folder, mode, now_date)
        make_prediction_video(cfg, output_folder, mode, now_date)
    elif mode == 'predict_image':
        make_prediction_image(cfg, output_folder, mode, now_date)
    elif mode == 'predict_video':
        make_prediction_video(cfg, output_folder, mode, now_date)
