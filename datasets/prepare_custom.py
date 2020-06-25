import os
import csv
import itertools
from detectron2.structures import BoxMode

DATASET_ROOT = './'
posture_file = 'posture_class.csv'
action_file = 'action_class.csv'
object_file = 'object_class.csv'


def get_custom_dataset(label, target):
    all_image = get_all_image_files(os.path.join(DATASET_ROOT, target))
    dataset_dicts = []
    labels = get_target_labels()[label]
    print(f' Prepare custom dataset - target: {target}, labels:{labels}')

    for image_id, image_file in enumerate(all_image):
        name, ext = os.path.splitext(image_file)
        csv_file = f'{name}-o.csv'

        with open(csv_file, "r", encoding='utf-8') as f:
            rdr = csv.reader(f, delimiter=',')
            lines = list(rdr)

            record = {}
            objs = []

            for idx, line in enumerate(lines):

                if idx == 0:
                    record["file_name"] = image_file
                    record["height"] = int(line[2])
                    record["width"] = int(line[1])
                    record["image_id"] = int(image_id)
                else:
                    xmin = int(line[0])
                    ymin = int(line[1])
                    xmax = int(line[0]) + int(line[2])
                    ymax = int(line[1]) + int(line[3])

                    poly = [
                        (xmin, ymin), (xmax, ymin),
                        (xmax, ymax), (xmin, ymax)
                    ]
                    poly = list(itertools.chain.from_iterable(poly))
                    obj = {
                        "bbox": [xmin, ymin, xmax, ymax],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": -1,  # temp
                        "iscrowd": 0
                    }
                    if label == 'multi':
                        multi = False
                        if len(line) > 5 and line[4] is not None and line[5] is not None:
                            multi = True
                            obj["category_id"] = 0
                            multi_id = 0
                            if line[4] in labels:
                                obj["category_id"] = labels.index(line[4])
                            if line[5] in labels:
                                multi_id = labels.index(line[5])
                            objs.append(obj)

                        if multi is True:
                            objs.append({
                                "bbox": [xmin, ymin, xmax, ymax],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": [poly],
                                "category_id": multi_id,  # temp
                                "iscrowd": 0
                            })
                    else:
                        if label == 'posture' and len(line) > 5 and line[4] is not None:
                            if line[4] in labels:
                                obj["category_id"] = labels.index(line[4])
                            else:
                                obj["category_id"] = 0
                            objs.append(obj)
                        elif label == 'action' and len(line) > 5 and line[5] is not None:
                            if line[5] in labels:
                                obj["category_id"] = labels.index(line[5])
                            else:
                                obj["category_id"] = 0
                            objs.append(obj)
                        elif label == 'object' and len(line) == 5 and line[4] is not None:
                            if line[4] in labels:
                                obj["category_id"] = labels.index(line[4])
                            else:
                                obj["category_id"] = 0
                            objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    # print(str(dataset_dicts))
    return dataset_dicts


def get_all_image_files(path):
    all_files = []
    exts = ['.png', '.jpeg', '.jpg']
    for (path, directory, files) in os.walk(path):
        for filename in files:
            file_abs_path = os.path.join(path, filename)
            ext = str(os.path.splitext(file_abs_path)[1]).lower()
            if ext in exts:
                all_files.append(str(file_abs_path))
    return all_files


def get_labels(path):
    class_list = []
    with open(path, "r") as f:
        rdr = csv.reader(f, delimiter=',')
        lines = list(rdr)
        if lines is not None and len(lines) > 0:
            class_list = lines[0]
    return class_list


def get_target_labels():
    return {
        'posture': get_labels(os.path.join(DATASET_ROOT, posture_file)),
        'action': get_labels(os.path.join(DATASET_ROOT, action_file)),
        'object': get_labels(os.path.join(DATASET_ROOT, object_file)),
        'multi': get_labels(os.path.join(DATASET_ROOT, posture_file)) + get_labels(os.path.join(DATASET_ROOT, action_file))
    }
