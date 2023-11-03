import os
import xml.etree.ElementTree as ET

class XMLToTXTConverter:
    def __init__(self, annotations_path, labels_path, classes):
        self.annotations_path = annotations_path
        self.labels_path = labels_path
        self.classes = classes

    def convert_annotation(self, annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        filename = root.find('filename').text.replace('.png', '.txt')
        image_size = root.find('size')
        image_width = int(image_size.find('width').text)
        image_height = int(image_size.find('height').text)

        label_file = os.path.join(self.labels_path, filename)

        with open(label_file, 'w') as file:
            for obj in root.iter('object'):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                x_center_norm = (xmin + xmax) / (2 * image_width)
                y_center_norm = (ymin + ymax) / (2 * image_height)
                width_norm = (xmax - xmin) / image_width
                height_norm = (ymax - ymin) / image_height

                file.write(
                    f"{self.classes[class_name]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
                )

    def convert_all(self):
        self.create_directory(self.labels_path)
        for annotation in os.listdir(self.annotations_path):
            annotation_file = os.path.join(self.annotations_path, annotation)
            self.convert_annotation(annotation_file)

    @staticmethod
    def create_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)


if __name__ == '__main__':
    annotations_path = "./Car-License-Plate/annotations"
    labels_path = "./Car-License-Plate/labels"
    classes = {"licence": 0}

    converter = XMLToTXTConverter(annotations_path, labels_path, classes)
    converter.convert_all()
