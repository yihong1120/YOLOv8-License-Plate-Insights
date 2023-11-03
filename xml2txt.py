import os
import xml.etree.ElementTree as ET
from typing import Dict

class XMLToTXTConverter:
    """
    A class to convert XML annotations to a TXT format suitable for object detection models.

    Attributes:
        annotations_path (str): Directory path where the XML files are stored.
        labels_path (str): Output directory path where the TXT files will be saved.
        classes (Dict[str, int]): A dictionary mapping class names to class ids.
    """

    def __init__(self, annotations_path: str, labels_path: str, classes: Dict[str, int]):
        """
        Initializes the XMLToTXTConverter object with paths and class mappings.

        Args:
            annotations_path (str): Directory path where the XML annotation files are stored.
            labels_path (str): Directory path where the resulting TXT files will be saved.
            classes (Dict[str, int]): A dictionary mapping class names to their corresponding IDs.
        """
        self.annotations_path: str = annotations_path
        self.labels_path: str = labels_path
        self.classes: Dict[str, int] = classes

    def convert_annotation(self, annotation_file: str) -> None:
        """
        Converts a single XML annotation file to a TXT format.

        Args:
            annotation_file (str): The path to the XML annotation file to be converted.
        """
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        filename: str = root.find('filename').text.replace('.png', '.txt')
        image_size = root.find('size')
        image_width: int = int(image_size.find('width').text)
        image_height: int = int(image_size.find('height').text)

        label_file: str = os.path.join(self.labels_path, filename)

        with open(label_file, 'w') as file:
            for obj in root.iter('object'):
                class_name: str = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin: int = int(bndbox.find('xmin').text)
                ymin: int = int(bndbox.find('ymin').text)
                xmax: int = int(bndbox.find('xmax').text)
                ymax: int = int(bndbox.find('ymax').text)

                x_center_norm: float = (xmin + xmax) / (2 * image_width)
                y_center_norm: float = (ymin + ymax) / (2 * image_height)
                width_norm: float = (xmax - xmin) / image_width
                height_norm: float = (ymax - ymin) / image_height

                file.write(
                    f"{self.classes[class_name]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
                )

    def convert_all(self) -> None:
        """
        Converts all XML annotation files found in the annotations_path to TXT format.
        """
        self.create_directory(self.labels_path)
        for annotation in os.listdir(self.annotations_path):
            annotation_file: str = os.path.join(self.annotations_path, annotation)
            self.convert_annotation(annotation_file)

    @staticmethod
    def create_directory(directory_path: str) -> None:
        """
        Creates a directory if it does not already exist.

        Args:
            directory_path (str): The path to the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)


if __name__ == '__main__':
    # Define paths to the annotations and labels directories and class mapping
    annotations_path: str = "./Car-License-Plate/annotations"
    labels_path: str = "./Car-License-Plate/labels"
    classes: Dict[str, int] = {"licence": 0}

    # Create an instance of the converter and convert all XML files to TXT format
    converter = XMLToTXTConverter(annotations_path, labels_path, classes)
    converter.convert_all()