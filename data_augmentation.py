import glob
import os
from typing import List, Tuple
import imageio.v2 as imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa


class DataAugmentation:
    """ 
    A class to perform data augmentation for image datasets, especially useful for training machine learning models.
    
    Attributes:
        train_path (str): The path to the training data.
        num_augmentations (int): The number of augmentations to perform per image.
        seq (iaa.Sequential): The sequence of augmentations to apply.
    """
    
    def __init__(self, train_path: str, num_augmentations: int = 1):
        """
        The constructor for DataAugmentation class.

        Parameters:
            train_path (str): The path to the training data.
            num_augmentations (int): The number of augmentations to perform per image.
        """
        self.train_path = train_path
        self.num_augmentations = num_augmentations
        # Define a sequence of augmentations
        self.seq = iaa.Sequential([
            # Various augmentations are defined here
        ], random_order=True)

    def augment_data(self):
        """ Performs the augmentation on the dataset. """
        image_paths = glob.glob(os.path.join(self.train_path, 'images', '*.png'))

        for image_path in image_paths:
            image = imageio.imread(image_path)
            # Replace 'images' with 'labels' in the path and change file extension to read labels
            label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
            image_shape = image.shape
            # Read bounding box information and initialise them on the image
            bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

            for i in range(self.num_augmentations):
                # If the image has an alpha channel, remove it
                if image.shape[2] == 4:
                    image = image[:, :, :3]

                # Perform augmentation
                image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)

                # Prepare filenames for augmented images and labels
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                aug_image_filename = f"{base_filename}_aug_{i}.png"
                aug_label_filename = f"{base_filename}_aug_{i}.txt"
                
                # Define paths for saving the augmented images and labels
                image_aug_path = os.path.join(self.train_path, 'images', aug_image_filename)
                label_aug_path = os.path.join(self.train_path, 'labels', aug_label_filename)

                # Save the augmented image and label
                imageio.imwrite(image_aug_path, image_aug)
                self.write_label_file(bbs_aug.remove_out_of_image().clip_out_of_image(), label_aug_path, image_shape[1], image_shape[0])

    @staticmethod
    def read_label_file(label_path: str, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        """
        Reads a label file and converts annotations into BoundingBox objects.

        Parameters:
            label_path (str): The path to the label file.
            image_shape (Tuple[int, int, int]): The shape of the image.

        Returns:
            List[BoundingBox]: A list of BoundingBox objects.
        """
        bounding_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    # Parsing label information for bounding box creation
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * image_shape[1]
                    y1 = (y_center - height / 2) * image_shape[0]
                    x2 = (x_center + width / 2) * image_shape[1]
                    y2 = (y_center + height / 2) * image_shape[0]
                    bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
        return bounding_boxes

    @staticmethod
    def write_label_file(bounding_boxes: List[BoundingBox], label_path: str, image_width: int, image_height: int):
        """
        Writes the augmented bounding box information back to a label file.

        Parameters:
            bounding_boxes (List[BoundingBox]): A list of augmented BoundingBox objects.
            label_path (str): The path where the label file is to be saved.
            image_width (int): The width of the image.
            image_height (int): The height of the image.
    """
        # Open the label file for writing
        with open(label_path, 'w') as f:
            # Iterate over the augmented bounding boxes and write them to the file
            for bb in bounding_boxes:
                # Calculate the center, width, and height for the YOLO format
                x_center = ((bb.x1 + bb.x2) / 2) / image_width
                y_center = ((bb.y1 + bb.y2) / 2) / image_height
                width = (bb.x2 - bb.x1) / image_width
                height = (bb.y2 - bb.y1) / image_height
                class_index = bb.label
                # Write the bounding box information in the YOLO format
                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':
    # Define the path to the training data and the number of augmentations per image
    train_path = 'dataset/train'
    num_augmentations = 15
    # Initialise the DataAugmentation class
    augmenter = DataAugmentation(train_path, num_augmentations)
    # Perform the data augmentation
    augmenter.augment_data()