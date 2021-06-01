import cv2
import argparse
import json
import numpy as np
import os
import xml.etree.ElementTree as ET


def create_image_annotation(file_name, width, height, image_id):
    file_name = os.path.basename(file_name)
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images


def create_annotation_yolo_format(min_x, min_y, max_x, max_y, image_id,
                                  category_id, annotation_id):
    bbox = (min_x, min_y, max_x, max_y)
    area = (max_x - min_x) * (max_y - min_y)

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': category_id,
        'segmentation': []
    }

    return annotation


# Create the annotations of the ECP dataset (Coco format)
coco_format = {"images": [{}], "categories": [], "annotations": [{}]}


# Get 'images' and 'annotations' info
def images_annotations_info(path, classes_mapping):

    # path : train.txt or test.txt
    annotations = []
    images = []

    image_id = 0  # Start image id with 1 too
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    for xml_file in path:

        tree = ET.parse(xml_file)

        h = int(tree.find("size/height").text)
        w = int(tree.find("size/width").text)
        filename = tree.find("filename").text

        objects = tree.findall("object")

        image = create_image_annotation(filename, w, h, image_id)
        images.append(image)

        for obj in objects:

            category_id = classes_mapping[obj.find("name").text]

            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)

            annotation = create_annotation_yolo_format(xmin, ymin, xmax, ymax,
                                                       image_id, category_id,
                                                       annotation_id)
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images, annotations


def convert_validation_data_to_coco(input_path: list, 
                                    output_file: str,
                                    labels: list):

    classes_mapping = {}
    for index, label in enumerate(labels):
        ann = {
            "supercategory": "None",
            "id": index + 1,  # Index starts with '1' .
            "name": label
        }
        coco_format['categories'].append(ann)
        classes_mapping[label] = index + 1

    # start converting format
    coco_format['images'], coco_format[
        'annotations'] = images_annotations_info(input_path, classes_mapping)

    with open(output_file, 'w+') as outfile:
        json.dump(coco_format, outfile)
