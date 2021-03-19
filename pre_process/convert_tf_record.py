from __future__ import division, print_function
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import sys

import cv2

XML_TEMPLATE = \
"""<annotation>
<folder/>
<filename>{}</filename>
<source>
<database>Unknown</database>
<annotation>Unknown</annotation>
<image>Unknown</image>
</source>
<size>
<width>{}</width>
<height>{}</height>
<depth>3</depth>
</size>
<segmented>0</segmented>
{}</annotation>
"""

OBJECT_TEMPLATE = \
"""<object>
<name>{}</name>
<occluded>0</occluded>
<bndbox>
<xmin>{}</xmin>
<ymin>{}</ymin>
<xmax>{}</xmax>
<ymax>{}</ymax>
</bndbox>
</object>
"""

def convert_example(set_type,
                    idx,
                    example,
                    images_path,
                    anno_txt_file,
                    new_size=None):
    """
    This converts each tensorflow example (a tensorflow example is one image and its associated annotations,
    called as such because the model learns from examples) into a format understood by yolov3
    """

    feature_map = {
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    features = tf.io.parse_single_example(example, feature_map)

    x_train = tf.image.decode_png(features['image/encoded'], channels=3)
    if new_size:
        x_train = tf.image.resize(x_train, new_size)

    img_width = features['image/width'].numpy()
    img_height = features['image/height'].numpy()
    img_fname = features['image/filename'].numpy().decode()
    img_encoded = features['image/encoded'].numpy()
    img_format = features['image/format'].numpy().decode()

    if img_format in ["jpeg", "jpg"]:
        img = tf.image.decode_jpeg(img_encoded, channels=3).numpy()
    elif img_format == 'png':
        img = tf.image.decode_png(img_encoded, channels=3).numpy()

    xmin = tf.sparse.to_dense(features['image/object/bbox/xmin']).numpy()
    ymin = tf.sparse.to_dense(features['image/object/bbox/ymin']).numpy()
    xmax = tf.sparse.to_dense(features['image/object/bbox/xmax']).numpy()
    ymax = tf.sparse.to_dense(features['image/object/bbox/ymax']).numpy()
    labels = [l.decode() for l in features['image/object/class/text'].values.numpy()]    

    objects_str = ""
    for c in list(zip(labels, xmin, ymin, xmax, ymax)):
        
        label = c[0]
        xmin = c[1]*img_width
        ymin = c[2]*img_height
        xmax = c[3]*img_width
        ymax = c[4]*img_height
        
        objects_str += OBJECT_TEMPLATE.format(label, xmin, ymin, xmax, ymax)

    full_xml = XML_TEMPLATE.format(img_fname, img_width, img_height, objects_str)
        
    with open(anno_txt_file.format(set_type, img_fname.split(".")[0]), "w+") as f:
        f.write(full_xml)

    img_path = images_path.format(set_type, img_fname)
    cv2.imwrite(img_path, img)

    return


def convert_tf_record(tf_records_path, output_path):

    tf.enable_eager_execution()

    images_path = os.path.join(output_path, "{}", "JPEGImages", "{}")
    anno_txt_file = os.path.join(output_path, "{}", "Annotations", "{}.xml")

    for dset in ["train", "val"]:
        dataset = tf.data.TFRecordDataset(tf_records_path.format(dset))
        for i, example in enumerate(dataset):
            convert_example(dset, i, example, images_path, anno_txt_file)
