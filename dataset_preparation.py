import os
import random
import shutil

class DatasetPreparation:
    def __init__(self, data_path, train_path, valid_path, split_ratio=0.8):
        self.data_path = data_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.split_ratio = split_ratio

    @staticmethod
    def create_dir(path):
        """Creates a new directory, deletes it first if it exists."""
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    @staticmethod
    def copy_data(files, source_dir, target_dir, file_type):
        """Copies files from source to target directory."""
        for file in files:
            source = os.path.join(source_dir, f'{file}{file_type}')
            target = os.path.join(target_dir, f'{file}{file_type}')
            if os.path.exists(source):
                shutil.copy(source, target)
            else:
                print(f"Warning: {source} does not exist.")

    def prepare(self):
        """Prepares the dataset directory structure and splits the dataset."""
        self.create_dir(os.path.join(self.train_path, 'images'))
        self.create_dir(os.path.join(self.train_path, 'labels'))
        self.create_dir(os.path.join(self.valid_path, 'images'))
        self.create_dir(os.path.join(self.valid_path, 'labels'))

        files = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.data_path, "images"))]
        random.shuffle(files)
        split_point = int(len(files) * self.split_ratio)

        train_files = files[:split_point]
        valid_files = files[split_point:]

        self.copy_data(train_files, os.path.join(self.data_path, "images"), os.path.join(self.train_path, "images"), '.png')
        self.copy_data(train_files, os.path.join(self.data_path, "labels"), os.path.join(self.train_path, "labels"), '.txt')
        self.copy_data(valid_files, os.path.join(self.data_path, "images"), os.path.join(self.valid_path, "images"), '.png')
        self.copy_data(valid_files, os.path.join(self.data_path, "labels"), os.path.join(self.valid_path, "labels"), '.txt')

        return train_files, valid_files

if __name__ == '__main__':
    data_path = 'Car-License-Plate'
    train_path = 'dataset/train'
    valid_path = 'dataset/valid'
    split_ratio = 0.8  # Adjust if needed

    dataset_preparation = DatasetPreparation(data_path, train_path, valid_path, split_ratio)
    train_files, valid_files = dataset_preparation.prepare()

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")
