import os
import json
import xml.etree.ElementTree as ET


def create_data_lists(data_path, output_folder, labels):
    label_map = {k: v + 1 for v, k in enumerate(labels)}
    label_map['background'] = 0  # Add the background class (necessary)

    train_images = list()
    train_objects = list()
    n_objects = 0

    annotation_ids = os.listdir(os.path.join(data_path, "Annotations"))
    image_ids = os.listdir(os.path.join(data_path, "JPEGImages"))

    assert len(annotation_ids) == len(image_ids)

    for a_id in annotation_ids:
        a_id = a_id.strip(".xml")

        # Parse annotation's XML file
        objects = parse_annotation(
            os.path.join(data_path, 'Annotations', a_id + '.xml'), label_map)
        if len(objects['boxes']) == 0:
            continue
        n_objects += len(objects)
        train_objects.append(objects)
        train_images.append(
            os.path.join(data_path, 'JPEGImages', a_id + '.jpg'))

    # Save to file
    if os.path.basename(data_path) == "train":
        with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
            json.dump(train_images, j)
        with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
            json.dump(train_objects, j)
        with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
            json.dump(label_map, j)  # save label map too
    else:
        with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
            json.dump(train_images, j)
        with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
            json.dump(train_objects, j)


def parse_annotation(annotation_path, label_map):
    tree = ET.parse(annotation_path)
    filename = tree.find("filename").text
    objects = tree.findall("object")
    height = tree.findall("height")
    width = tree.findall("height")

    boxes = list()
    labels = list()
    difficulties = list()
    
    for obj in objects:

        difficult = 0 # Always consider an object not difficult
        label = obj.find("name").text
        if label not in label_map.keys():
            raise Exception(f"Label {label} found in annotation {annotation_path} but doesn't exist in label map.")

        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        # Error checking
        problem_files = [f"Bad value {i[0]}: {i[1]} in {annotation_path}" \
            for i in [("xmin", xmin), ("ymin", ymin), ("xmax", xmax), ("ymax"), ymax] if i <= 0] # TODO: width/height check?

        if xmax >= xmin:
            problem_files.append(f"xmax is greater than or equal to xmin in {annotation_path}")
        if ymax >= ymin:
            problem_files.append(f"ymax is greater than or equal to ymin in {annotation_path}")

        if len(problem_files):
            [print(p + ", skipping") for p in problem_files]
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}
