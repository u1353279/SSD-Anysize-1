import argparse
import numpy as np
import xml.etree.ElementTree as ET

def comb_for_labels(annotations_files_list):

    classes = set()
    class_distribution = dict()

    for xml_file in annotations_files_list:

        tree = ET.parse(xml_file)

        h = int(tree.find("size/height").text)
        w = int(tree.find("size/width").text)
        filename = tree.find("filename").text

        objects = tree.findall("object")

        for obj in objects:
            if obj.find("name").text not in classes:
                classes.add(obj.find("name").text)
                class_distribution[obj.find("name").text] = 1
            else:
                class_distribution[obj.find("name").text] += 1

    return classes, class_distribution