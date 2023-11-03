import os
import random
import shutil
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio.v2 as imageio

def create_dir(path):
    """創建新目錄，如果目錄存在則先刪除"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def copy_data(files, source_dir, target_dir, file_type):
    """將文件從源目錄複製到目標目錄"""
    for file in files:
        source = os.path.join(source_dir, f'{file}{file_type}')
        target = os.path.join(target_dir, f'{file}{file_type}')
        if os.path.exists(source):
            shutil.copy(source, target)
        else:
            print(f"Warning: {source} does not exist.")

def prepare_dataset(data_path, train_path, valid_path, split_ratio=0.8):
    """準備數據集目錄結構並分割數據集"""
    create_dir(os.path.join(train_path, 'images'))
    create_dir(os.path.join(train_path, 'labels'))
    create_dir(os.path.join(valid_path, 'images'))
    create_dir(os.path.join(valid_path, 'labels'))

    files = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_path, "images"))]
    random.shuffle(files)
    split_point = int(len(files) * split_ratio)

    train_files = files[:split_point]
    valid_files = files[split_point:]

    copy_data(train_files, os.path.join(data_path, "images"), os.path.join(train_path, "images"), '.png')
    copy_data(train_files, os.path.join(data_path, "labels"), os.path.join(train_path, "labels"), '.txt')
    copy_data(valid_files, os.path.join(data_path, "images"), os.path.join(valid_path, "images"), '.png')
    copy_data(valid_files, os.path.join(data_path, "labels"), os.path.join(valid_path, "labels"), '.txt')

    return train_files, valid_files


# 定義一個函數來解析標籤文件
def read_label_file(label_path):
    bbs = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            # 假設標籤文件中每行的格式為: class x_center y_center width height
            # 這需要根據你的標籤文件具體格式進行調整
            c, x_center, y_center, width, height = map(float, line.split())
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=c))
    return bbs

# 定義一個函數來寫入更新後的標籤文件
def write_label_file(bbs, label_path):
    # Ensure that the directory where the label_path points to exists
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    
    with open(label_path, 'w') as f:
        for bb in bbs.bounding_boxes:
            x_center = (bb.x1 + bb.x2) / 2
            y_center = (bb.y1 + bb.y2) / 2
            width = bb.x2 - bb.x1
            height = bb.y2 - bb.y1
            # 再次假設每行格式為: class x_center y_center width height
            f.write(f"{int(bb.label)} {x_center} {y_center} {width} {height}\n")

def data_augmentation(train_path):
    # 定義增強序列
    seq = iaa.Sequential([
        iaa.Flipud(0.5), # 垂直翻轉圖像和標籤
        iaa.Fliplr(0.5), # 水平翻轉圖像和標籤
        iaa.Affine(rotate=(-25, 25)) # 旋轉圖像和標籤
    ])

    augmented_images_path = os.path.join(train_path, 'augmented_images')
    
    # Create the directory for augmented images if it does not exist
    if not os.path.exists(augmented_images_path):
        os.makedirs(augmented_images_path)

    image_paths = [os.path.join(train_path, 'images', file) for file in os.listdir(os.path.join(train_path, 'images')) if file.endswith('.png')]
    for image_path in image_paths:
        # Read the image using imageio
        image = imageio.imread(image_path)
        bbs = BoundingBoxesOnImage(read_label_file(image_path.replace('images', 'labels').replace('.png', '.txt')), shape=image.shape)
        
        # 增強圖像和標籤
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        
        # 保存增強後的圖像和標籤
        image_aug_path = image_path.replace('images', 'augmented_images')
        label_aug_path = image_aug_path.replace('images', 'labels').replace('.png', '.txt')
        # Generate the full path for the augmented image
        image_filename = os.path.basename(image_path)
        image_aug_path = os.path.join(augmented_images_path, 'aug_' + image_filename)
        
        # Save the augmented image
        imageio.imwrite(image_aug_path, image_aug)

        write_label_file(bbs_aug.remove_out_of_image().clip_out_of_image(), label_aug_path)

# 主函數
def main():
    data_path = 'Car-License-Plate'
    train_path = 'dataset/train'
    valid_path = 'dataset/valid'

    # 準備數據集
    train_files, valid_files = prepare_dataset(data_path, train_path, valid_path)

    # 調用數據增強函數
    data_augmentation(train_path)

if __name__ == '__main__':
    main()
