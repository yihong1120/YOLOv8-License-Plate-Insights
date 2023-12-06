import argparse
import os
from defusedxml import ElementTree as ET
from typing import Dict
from tqdm import tqdm


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
        Initialises the XMLToTXTConverter object with paths and class mappings.

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
        # 获取所有注释文件并使用tqdm进行迭代
        annotation_files = os.listdir(self.annotations_path)
        for annotation in tqdm(annotation_files, desc="Converting XML to TXT"):
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

def parse_classes(classes_str: str) -> Dict[str, int]:
    """
    Parses the class string into a dictionary.

    Args:
        classes_str (str): A string containing class mappings in the 'class1=id1,class2=id2,...' format.

    Returns:
        Dict[str, int]: A dictionary mapping class names to their corresponding IDs.
    """
    class_pairs = classes_str.split(',')
    return {pair.split('=')[0]: int(pair.split('=')[1]) for pair in class_pairs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XML annotations to TXT format for object detection models.')

    parser.add_argument('--annotations_path', type=str, default='./Car-License-Plate/annotations', help='Directory path where the XML files are stored.')
    parser.add_argument('--labels_path', type=str, default='./Car-License-Plate/labels', help='Output directory path where the TXT files will be saved.')
    parser.add_argument('--classes', type=str, default='licence=0', help='Comma-separated list of class mappings in the format "class1=id1,class2=id2,...".')
    
    args = parser.parse_args()

    classes: Dict[str, int] = parse_classes(args.classes)

    # Create an instance of the converter and convert all XML files to TXT format
    converter = XMLToTXTConverter(args.annotations_path, args.labels_path, classes)
    converter.convert_all()